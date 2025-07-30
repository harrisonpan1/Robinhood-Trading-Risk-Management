import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime

from login import Robinhood
from robin_stocks.robinhood import stocks, options, helper


def adj_uvix_uvix3(df):
    uvix_map = (
        df[df["chain_symbol"] == "UVIX"]
        .set_index(["expiration_date", "strike_price", "option_type"])["option_id"]
        .to_dict()
    )

    def get_correct_id(row):
        # Create a key from the row's properties
        lookup_key = (row["expiration_date"], row["strike_price"], row["option_type"])
        # Return the new ID from the map, or the original ID if not found
        return uvix_map.get(lookup_key, row["option_id"])

    # --- 4. Apply the function to the 'UVIX3' rows ---
    # Identify the rows that need to be updated
    uvix3_mask = df["chain_symbol"] == "UVIX3"

    # Apply the function ONLY to those rows and update the 'option_id' column
    df.loc[uvix3_mask, "option_id"] = df.loc[uvix3_mask].apply(get_correct_id, axis=1)

    # You can also update the 'chain_symbol' at the same time if you wish
    df.loc[uvix3_mask, "chain_symbol"] = "UVIX"
    return df


def stock_order_journal(df, start_date=None, end_date=None):
    """
    Calculates gains and losses from a DataFrame of stock orders,
    accounting for option exercises, assignments, and stock splits.

    Args:
        df (pd.DataFrame):
            A DataFrame containing stock order history. Expected columns are:
            'symbol', 'date', 'side', 'quantity', 'average_price', 'fees'.
        start_date (str, optional):
            The start date for filtering orders in ISO8601 format (YYYY-MM-DD).
        end_date (str, optional):
            The end date for filtering orders in ISO8601 format (YYYY-MM-DD).

    Returns:
        pd.DataFrame:
            A DataFrame detailing each sell transaction with its calculated
            average buy price, profit/loss, and total gain. Returns an empty
            DataFrame if no sell transactions can be processed.
    """
    # --- Data Preparation and Augmentation ---
    print("--- Starting Trade Journal Calculation ---")
    df_processed = df.copy()
    # Standardize column names and ensure correct data types
    df_processed = df_processed.rename(columns={"date": "timestamp"})
    df_processed["timestamp"] = pd.to_datetime(
        df_processed["timestamp"], format="ISO8601"
    )
    df_processed["quantity"] = df_processed["quantity"].astype(float)
    df_processed["average_price"] = df_processed["average_price"].astype(float)
    df_processed["fees"] = df_processed["fees"].astype(float).fillna(0.0)

    # Apply date filters if provided
    if start_date:
        start_date_dt = pd.to_datetime(start_date).tz_localize(
            df_processed["timestamp"].dt.tz
        )
        df_processed = df_processed[df_processed["timestamp"] >= start_date_dt]
        print(f"Filtering orders from {start_date} onwards.")
    if end_date:
        end_date_dt = pd.to_datetime(end_date).tz_localize(
            df_processed["timestamp"].dt.tz
        )
        df_processed = df_processed[df_processed["timestamp"] <= end_date_dt]
        print(f"Filtering orders up to {end_date}.")

    # adj UVIX split
    # Adjust for UVIX split: for all rows with symbol 'UVIX' and timestamp <= 2025-01-25, apply a split 1/10 adjustment
    split_date = pd.Timestamp("2025-01-25", tz="UTC")
    uvix_mask = (df_processed["symbol"] == "UVIX") & (
        df_processed["timestamp"] <= split_date
    )
    split_ratio = 1 / 10
    new_prices = df_processed.loc[uvix_mask, "average_price"] / split_ratio
    new_quantities = df_processed.loc[uvix_mask, "quantity"] * split_ratio
    df_processed.loc[uvix_mask, "average_price"] = new_prices
    df_processed.loc[uvix_mask, "quantity"] = new_quantities

    all_symbols = np.append(
        df_processed["symbol"].unique(), "SOXS"
    )  # need a more robust way to get all symbols
    option_event_orders = []

    print(
        "\n--- Step 1: Checking for option exercise/assignment events from Robinhood ---"
    )
    for symbol in all_symbols:
        print(f"Processing symbol: {symbol}")
        try:
            # This makes a live API call to Robinhood
            events = stocks.get_events(symbol)
            for event in events:
                if event["state"] != "confirmed":
                    continue
                if start_date:
                    if event["created_at"] < start_date:
                        continue
                if end_date:
                    if event["created_at"] > end_date:
                        continue
                # Process events that result in stock transactions
                if event.get("type") in ["assignment", "exercise"]:
                    for component in event.get("equity_components", []):
                        # Create a new order record from the event data
                        new_order = {
                            "symbol": component["symbol"],
                            "timestamp": pd.to_datetime(event["created_at"]),
                            "order_type": event["type"],
                            "side": component["side"],
                            "fees": 0.0,
                            "quantity": float(component["quantity"]),
                            "average_price": float(component["price"]),
                        }
                        option_event_orders.append(new_order)
                        print(
                            f"  -> Found '{event['type']}' event for {symbol}: "
                            f"{component['side']} {component['quantity']} @ {component['price']}"
                        )
        except Exception as e:
            print(f"  Could not fetch events for {symbol}: {e}")

    if option_event_orders:
        events_df = pd.DataFrame(option_event_orders)
        # Combine original orders with the new orders from option events
        augmented_df = pd.concat([df_processed, events_df], ignore_index=True)
        print("Successfully added orders from option events to the journal.")
    else:
        augmented_df = df_processed
        print("No relevant option events found.")

    # --- Process Trades and Identify Open Positions ---
    journal_entries, open_positions = [], []
    for symbol in augmented_df["symbol"].unique():
        symbol_trades = augmented_df[augmented_df["symbol"] == symbol]
        buys = symbol_trades[symbol_trades["side"] == "buy"].sort_values(
            by="timestamp", ascending=False
        )
        sells = symbol_trades[symbol_trades["side"] == "sell"].sort_values(
            by="timestamp"
        )

        if sells.empty:
            open_positions.extend(buys.to_dict("records"))
            continue

        if buys.empty:
            open_positions.extend(sells.to_dict("records"))
            continue

        # --- Case C: Both Buys and Sells (FIFO Matching) ---
        buy_queue = deque(
            buys[["quantity", "average_price", "timestamp"]].values.tolist()
        )
        for sell_order in sells.itertuples():
            sell_qty_remaining = sell_order.quantity
            avg_price_info = {"qty": [], "price": []}
            while sell_qty_remaining > 1e-8:
                if not buy_queue:
                    open_positions.append(sell_order._asdict())
                    break

                # CHANGE 3: Take from the front of the queue (oldest buy)
                buy_pop = buy_queue.popleft()
                buy_qty, buy_price, buy_date = buy_pop
                match_qty = min(sell_qty_remaining, buy_qty)
                avg_price_info["qty"].append(match_qty)
                avg_price_info["price"].append(buy_price)
                sell_qty_remaining -= match_qty
                remaining_buy_qty = buy_qty - match_qty
                if remaining_buy_qty > 1e-8:
                    # CHANGE 4: Put the remainder back at the front
                    buy_queue.appendleft([remaining_buy_qty, buy_price, buy_date])
            if not avg_price_info["qty"]:
                continue
            entry = {
                "close_date": sell_order.timestamp,
                "symbol": symbol,
                "quantity": sell_order.quantity,
                "sellPrice": sell_order.average_price,
                "buyPrice": np.average(
                    avg_price_info["price"], weights=avg_price_info["qty"]
                ),
                "fee": sell_order.fees,
            }
            journal_entries.append(entry)

        # Any buys left in the queue are open long positions
        for leftover_buy in buy_queue:
            open_pos_record = {
                "symbol": symbol,
                "quantity": leftover_buy[0],
                "average_price": leftover_buy[1],
                "timestamp": leftover_buy[2],
                "side": "buy",
            }
            open_positions.append(open_pos_record)

    # --- 3. Finalize and Return DataFrames ---
    journal_df = pd.DataFrame(journal_entries)
    if not journal_df.empty:
        journal_df["total_gain"] = (
            journal_df["sellPrice"] - journal_df["buyPrice"]
        ) * journal_df["quantity"] - journal_df["fee"]
        journal_df = journal_df.sort_values(by="close_date", ascending=False)

    open_positions_df = pd.DataFrame(open_positions)
    if not open_positions_df.empty:
        open_positions_df = open_positions_df.sort_values(
            by="timestamp", ascending=False
        )

    return journal_df, open_positions_df


def option_order_journal(df_orders, start_date=None, end_date=None):
    """
    Calculates realized gains and losses from an options order book,
    correctly handling expirations based on the current date.

    Args:
        df_orders (pd.DataFrame): DataFrame containing the options order history.
        start_date (str, optional): Start date for filtering orders in 'YYYY-MM-DD' format.
        end_date (str, optional): End date for filtering orders in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A journal of all closed option trades with P/L.
    """
    # Prepare the DataFrame by renaming columns to match the journal logic
    df_prepared = df_orders.rename(
        columns={
            "order_created_at": "timestamp",
            "expiration_date": "expDate",
            "chain_symbol": "symbol",
            "processed_quantity": "quantity",
        }
    )
    # Convert price from per-share to per-contract (x100)
    df_prepared["price"] = df_prepared["price"] * 100
    df_prepared["expDate"] = pd.to_datetime(df_prepared["expDate"])

    journal_entries = []
    open_positions = []

    # --- Group by option_id ---
    for option_id, group in df_prepared.groupby("option_id"):

        # Split into buys and sells *within the group*
        buys = (
            group[group["side"] == "buy"]
            .copy()
            .sort_values(by="timestamp", ascending=False)
        )
        sells = group[group["side"] == "sell"].copy().sort_values(by="timestamp")

        # Case A: Only Sells in this group (a short position)
        if buys.empty:
            sell_details = sells.iloc[0]  # Get common details from the first record

            if sell_details["expDate"].date() < datetime.now().date():
                # Aggregate all sell orders in this group
                total_quantity_sold = sells["quantity"].sum()
                avg_sell_price = np.average(sells["price"], weights=sells["quantity"])
                total_fee = sells["fees"].sum()

                entry = {
                    "expDate": sell_details["expDate"].strftime("%Y/%m/%d"),
                    "symbol": sell_details["symbol"],
                    "strike": sell_details["strike_price"],
                    "option_type": sell_details["option_type"],
                    "quantity": total_quantity_sold,  # Use aggregated quantity
                    "sellPrice": avg_sell_price,  # Use aggregated price
                    "buyPrice": 0,
                    "option_id": option_id,
                    "fee": total_fee,  # Aggregate fees
                }
                journal_entries.append(entry)
            else:
                # It's an open position.
                open_positions.extend(sells.to_dict("records"))
            continue

        # Case B: Only Buys in this group (a long position)
        if sells.empty:
            buy_details = buys.iloc[0]  # Get common details from the first record
            if buy_details["expDate"].date() < datetime.now().date():
                # Aggregate all buy orders in this group
                total_quantity_bought = buys["quantity"].sum()
                avg_buy_price = np.average(buys["price"], weights=buys["quantity"])
                total_fee = buys["fees"].sum()

                entry = {
                    "expDate": buy_details["expDate"].strftime("%Y/%m/%d"),
                    "symbol": buy_details["symbol"],
                    "strike": buy_details["strike_price"],
                    "option_type": buy_details["option_type"],
                    "quantity": total_quantity_bought,  # Use aggregated quantity
                    "quantity": avg_buy_price,  # Use aggregated price
                    "sellPrice": 0,
                    "option_id": option_id,
                    "fee": total_fee,  # Aggregate fees
                }
                journal_entries.append(entry)
            else:
                # It's an open position.
                open_positions.extend(buys.to_dict("records"))
            continue

        # Case C: Both Buys and Sells in this group (a closed position)
        buy_stack = buys.to_dict("records")
        for sell_record in sells.to_dict("records"):
            sell_qty_remaining = sell_record["quantity"]
            avg_price_info = {"qty": [], "price": [], "fees": []}

            while sell_qty_remaining > 1e-6:  # Use tolerance for float comparison
                if not buy_stack:
                    # This sell has no corresponding buys left, so it opens a short position
                    # We'll add it to the open positions list if not expired
                    if (
                        pd.to_datetime(sell_record["expDate"]).date()
                        >= datetime.now().date()
                    ):
                        sell_record["quantity"] = (
                            sell_qty_remaining  # Update quantity to remaining
                        )
                        open_positions.append(sell_record)
                    break  # Exit the while loop

                buy_record = buy_stack.pop()
                qty_to_match = min(sell_qty_remaining, buy_record["quantity"])

                # Collect data for weighted average calculation
                avg_price_info["qty"].append(qty_to_match)
                avg_price_info["price"].append(buy_record["price"])
                avg_price_info["fees"].append(buy_record["fees"])

                # If buy is partially used, put the remainder back on the stack
                buy_record["quantity"] -= qty_to_match
                if buy_record["quantity"] > 1e-6:
                    buy_stack.append(buy_record)

                sell_qty_remaining -= qty_to_match

            # If no match was made, continue to next sell record
            if not avg_price_info["qty"]:
                continue

            # Create the journal entry for this closed trade
            avg_buy_price = np.average(
                avg_price_info["price"], weights=avg_price_info["qty"]
            )
            total_buy_fees = sum(avg_price_info["fees"])
            total_fee = sell_record["fees"] + total_buy_fees

            entry = {
                "expDate": sell_record["expDate"].strftime("%Y/%m/%d"),
                "symbol": sell_record["symbol"],
                "strike": sell_record["strike_price"],
                "option_type": sell_record["option_type"],
                "quantity": sell_record["quantity"],
                "sellPrice": sell_record["price"],
                "buyPrice": avg_buy_price,
                "option_id": option_id,
                "fee": total_fee,
            }
            journal_entries.append(entry)

        # Handle any leftover buys in the stack (they are open or expired)
        for buy_record in buy_stack:
            if pd.to_datetime(buy_record["expDate"]).date() < datetime.now().date():
                entry = {
                    "expDate": pd.to_datetime(buy_record["expDate"]).strftime(
                        "%Y/%m/%d"
                    ),
                    "symbol": buy_record["symbol"],
                    "strike": buy_record["strike_price"],
                    "option_type": buy_record["option_type"],
                    "quantity": buy_record["quantity"],
                    "buyPrice": buy_record["price"],
                    "sellPrice": 0,
                    "option_id": option_id,
                    "fee": buy_record["fees"],
                }
                journal_entries.append(entry)
            else:
                open_positions.append(buy_record)

    # Final step: filter journal entries by date if provided
    if start_date or end_date:
        journal_df = pd.DataFrame(journal_entries)
        journal_df["close_date"] = pd.to_datetime(
            journal_df["expDate"]
        )  # Assuming expDate is the close date
        if start_date:
            journal_df = journal_df[
                journal_df["close_date"] >= pd.to_datetime(start_date)
            ]
        if end_date:
            journal_df = journal_df[
                journal_df["close_date"] <= pd.to_datetime(end_date)
            ]
        journal_entries = journal_df.to_dict("records")

    return pd.DataFrame(journal_entries).sort_values(
        by="expDate", ascending=False
    ), pd.DataFrame(open_positions).sort_values(by="expDate", ascending=False)


def calculate_summary(group):
    """
    Calculates the total quantity and weighted average price for a group.
    """
    total_quantity = group["quantity"].sum()
    # Ensure there's a quantity to avoid division by zero
    if total_quantity > 0:
        avg_price = np.average(group["average_price"], weights=group["quantity"])
    else:
        avg_price = 0

    return pd.Series({"quantity": total_quantity, "avg_price": avg_price})


def sign_from_side(row):
    # For options, 'buy' is long, 'sell' is short
    return 1 if row["type"] == "long" else -1


def greeks_manager(rh_obj):
    """
    Aggregates the Greeks for all open positions.
    """
    df_open_stock_position = rh_obj.get_open_stock_positions()
    df_open_option_position = rh_obj.get_open_options_positions()

    # get option greeks
    def fetch_greeks(row):
        greeks = rh_obj.get_option_greeks(row["option_id"])[0]
        option_info = helper.request_get(greeks["instrument"])
        # If greeks is None or missing keys, handle gracefully
        return pd.Series(
            {
                "symbol": option_info.get("chain_symbol") if option_info else None,
                "expiration": (
                    option_info.get("expiration_date") if option_info else None
                ),
                "strike": option_info.get("strike_price") if option_info else None,
                "delta": greeks.get("delta") if greeks else None,
                "gamma": greeks.get("gamma") if greeks else None,
                "theta": greeks.get("theta") if greeks else None,
                "vega": greeks.get("vega") if greeks else None,
                "rho": greeks.get("rho") if greeks else None,
            }
        )

    df_open_option_position[
        ["symbol", "expiration", "strike", "delta", "gamma", "theta", "vega", "rho"]
    ] = df_open_option_position.apply(fetch_greeks, axis=1)
    df_greeks = df_open_option_position[
        [
            "symbol",
            "expiration",
            "strike",
            "type",
            "quantity",
            "delta",
            "gamma",
            "theta",
            "vega",
            "rho",
        ]
    ]

    greek_cols = ["delta", "gamma", "theta", "vega", "rho"]
    # Convert greek columns to float (they may be object dtype)
    for col in greek_cols:
        df_greeks[col] = pd.to_numeric(df_greeks[col], errors="coerce")
    df_greeks["quantity"] = pd.to_numeric(df_greeks["quantity"], errors="coerce")
    df_greeks["sign"] = df_greeks.apply(sign_from_side, axis=1)

    def agg_greeks(group):
        result = {}
        for col in greek_cols:
            result[f"total_{col}"] = (
                group[col] * 100 * group["quantity"] * group["sign"]
            ).sum()
        return pd.Series(result)

    agg_result = df_greeks.groupby("symbol").apply(agg_greeks).reset_index()

    # add stock delta to agg_result
    stock_deltas = {}
    for symbol in df_open_stock_position["symbol"].unique():
        row = df_open_stock_position[df_open_stock_position["symbol"] == symbol]
        sign = 1 if row["type"].values[0] == "long" else -1
        quantity = row["quantity"].values[0]
        stock_deltas[symbol] = sign * quantity

    agg_result["total_delta"] = agg_result.apply(
        lambda row: (
            row["total_delta"] + float(stock_deltas[row["symbol"]])
            if row["symbol"] in stock_deltas
            else row["total_delta"]
        ),
        axis=1,
    )

    return agg_result


if __name__ == "__main__":
    print("*_" * 75)
    print("Starting the trading journal script.")

    # --- Step 1: Create an instance of the class to log in ---
    rh = Robinhood()

    # --- Step 2: Use the class methods if login was successful ---
    if rh.login_successful:
        print("Login successful.")

        # 1. export option orders and stock orders
        rh.export_stock_orders(dir_path="./data")
        print("\n--- Stock orders exported successfully ---")

        # 2. test export_option_orders
        rh.export_option_orders(dir_path="./data")
        print("\n--- Option orders exported successfully ---")

        # 3. Create the stock and option journals
        print("\n--- Creating stock and option journals ---")
        # Load the stock orders DataFrame
        stock_orders_df = pd.read_csv("./data/stock_orders.csv")
        journal_df, open_positions_df_agg = stock_order_journal(stock_orders_df)
        # Save the stock journal DataFrame to a CSV file
        journal_df.to_csv("./data/stock_journal.csv", index=False)
        open_positions_df_agg.to_csv("./data/stock_open_positions.csv", index=False)
        print("\n--- Stock order journal created successfully ---")

        # Load the option orders DataFrame
        option_orders_df = pd.read_csv("./data/option_orders.csv")
        # Adjust UVIX and UVIX3 option IDs
        option_orders_df = adj_uvix_uvix3(option_orders_df)
        option_journal_df, open_positions_df = option_order_journal(
            option_orders_df, start_date="2024-01-01"
        )
        # Save the option journal DataFrame to a CSV file
        option_orders_df.to_csv("./data/option_orders.csv", index=False)
        option_journal_df.to_csv("./data/option_journal.csv", index=False)
        open_positions_df.to_csv("./data/option_open_positions.csv", index=False)
        print("\n--- Option order journal created successfully ---")

        # 4. Get the aggregated greeks for all open positions
        print("\n--- Fetching aggregated greeks for all open positions ---")
        df_greeks = greeks_manager(rh)
        # Save the aggregated greeks DataFrame to a CSV file
        df_greeks.to_csv("./data/greeks.csv", index=False)
        print("\n--- Aggregated greeks for all open positions saved successfully ---")

        # Log out when you are finished
        rh._logout()
