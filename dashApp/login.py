import robin_stocks.robinhood as r
from robin_stocks.robinhood import stocks, options, helper
import getpass
import pandas as pd
from datetime import datetime
import pathlib as path
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")


class Robinhood:
    def __init__(self, username=USERNAME, password=PASSWORD):
        """
        Initializes and logs into the Robinhood account upon creation of the object.
        """
        self.instrument_cache = {}
        self.option_instrument_cache = {}
        self.login_successful = self._login(username, password)
        if not self.login_successful:
            print("Initialization failed due to login error.")

    def _login(self, username, password):
        """Handles the login process."""
        if not username:
            username = input("Enter your Robinhood username/email: ")
        if not password:
            password = getpass.getpass("Enter your Robinhood password: ")

        print("\nAttempting to log in...")
        print("Please check your phone for any approval notifications.")

        try:
            login_data = r.login(username, password)
            if "access_token" in login_data:
                print("✅ Successfully logged in!")
                return True
            else:
                print(f"\n❌ Login failed. Response: {login_data}")
                return False
        except Exception as e:
            print(f"\n❌ An unexpected error occurred during login: {e}")
            return False

    def _logout(self):
        """Logs out of the Robinhood session."""
        if self.login_successful:
            r.logout()
            print("\n✅ Successfully logged out.")
            self.login_successful = False

    def get_instrumentBasic(self, url):
        """
        Accepts a single instrument URL or a list of URLs and returns basic instrument data,
        matching the name and functionality of your original method.

        :param url: A single URL string or a list of URL strings.
        :return: A dictionary for a single URL or a pandas DataFrame for a list of URLs.
        """
        if not self.login_successful:
            return "Not logged in."

        is_single_item = isinstance(url, str)
        urls = [url] if is_single_item else url

        results = []
        for current_url in urls:
            if current_url in self.instrument_cache:
                results.append(self.instrument_cache[current_url])
                continue

            try:
                instrument_data = stocks.get_instrument_by_url(current_url)
                if instrument_data:
                    latest_price = stocks.get_latest_price(instrument_data["symbol"])[0]
                    # split_data = stocks.get_splits(instrument_data["symbol"])[0]
                    formatted_instrument = {
                        "instrument_id": instrument_data.get("id"),
                        "url": instrument_data.get("url"),
                        "state": instrument_data.get("state"),
                        "market": helper.request_get(instrument_data["market"]).get(
                            "name", "N/A"
                        ),
                        "name": instrument_data.get("name"),
                        "symbol": instrument_data.get("symbol"),
                        "country": instrument_data.get("country_code"),
                        "instrumentType": instrument_data.get("type"),
                        # "splits": split_data,
                        "quote": latest_price,
                        "quote_time": datetime.now(),
                    }
                    results.append(formatted_instrument)
                    self.instrument_cache[current_url] = formatted_instrument
            except Exception as e:
                print(f"Error processing URL {current_url}: {e}")

        if is_single_item:
            return results[0] if results else None
        else:
            return pd.DataFrame(results).drop_duplicates()

    def get_option_instrumentBasic(self, url):
        """
        Fetches and caches details for a specific option contract by its URL.
        """
        if not self.login_successful:
            return {}

        # Use cache to avoid repeated API calls
        if (
            hasattr(self, "option_instrument_cache")
            and url in self.option_instrument_cache
        ):
            return self.option_instrument_cache[url]

        try:
            # Use the robin-stocks helper to get data from the URL
            response = helper.request_get(url)
            if not response:
                return {}

            exp_date_obj = datetime.strptime(
                response["expiration_date"], "%Y-%m-%d"
            ).date()
            strike_price = float(response["strike_price"])
            option_type_str = response["type"].upper()

            # Create a human-readable name using an f-string
            option_name = f"{response['chain_symbol']} ${strike_price:.2f} {option_type_str} {exp_date_obj}"

            formatted_data = {
                "optionName": option_name,
                "optionId": response.get("id"),
                "state": response["state"],
                "strikePrice": response["strike_price"],
                "optionType": option_type_str,
                "expDate": exp_date_obj,
            }

            # Cache the result before returning
            if not hasattr(self, "option_instrument_cache"):
                self.option_instrument_cache = {}
            self.option_instrument_cache[url] = formatted_data

            return formatted_data

        except Exception as e:
            print(f"Error fetching option object for {url}: {e}")
            return {}

    def get_all_stock_orders(self):
        """
        Fetches all stock positions (open and closed) and returns a pandas DataFrame.
        """
        if not self.login_successful:
            return "Not logged in."
        orders = r.get_all_stock_orders()
        return orders if orders else []

    # def get_all_stock_positions(self):  # no need
    #     """
    #     Fetches all historical stock orders and returns a pandas DataFrame.
    #     """
    #     if not self.login_successful:
    #         return "Not logged in."
    #     positions = r.get_all_stock_orders()
    #     if not positions:
    #         return pd.DataFrame()

    #     for pos in positions:
    #         instrument_data = r.get_instrument_by_url(pos["instrument"])
    #         pos["symbol"] = instrument_data.get("symbol", "N/A")

    #     return pd.DataFrame(positions)

    def get_open_stock_positions(self):
        """
        Fetches all open stock positions and returns a pandas DataFrame.
        """
        if not self.login_successful:
            return "Not logged in."
        positions = r.get_open_stock_positions()
        if not positions:
            return pd.DataFrame()

        for pos in positions:
            instrument_data = r.get_instrument_by_url(pos["instrument"])
            pos["symbol"] = instrument_data.get("symbol", "N/A")

        return pd.DataFrame(positions)

    def get_all_options_orders(self):
        """
        Fetches all historical option orders and returns a pandas DataFrame.
        """
        if not self.login_successful:
            return "Not logged in."
        orders = r.get_all_option_orders()
        return orders if orders else []

    def get_open_options_positions(self):
        """
        Fetches all open option positions and returns a pandas DataFrame.
        """
        if not self.login_successful:
            return "Not logged in."
        options = r.get_open_option_positions()
        return pd.DataFrame(options) if options else pd.DataFrame()

    def get_quotes(self, symbols):
        """
        Gets the latest price for one or more stock symbols.

        :param symbols: A single string or a list of stock ticker strings.
        :return: A dictionary of {symbol: price} or a single price string.
        """
        if not self.login_successful:
            return "Not logged in."
        prices = r.get_latest_price(symbols, includeExtendedHours=False)
        # The library returns a list with None for invalid tickers; let's clean it up.
        if isinstance(symbols, list):
            return {
                symbol: price
                for symbol, price in zip(symbols, prices)
                if price is not None
            }
        return prices[0]  # Return the single price string if only one symbol was passed

    def get_option_greeks(self, option_id):
        """
        Fetches Greeks for a specific option contract by its ID.

        :param option_id: The ID of the option contract.
        :return: A dictionary containing the Greeks or an error message.
        """
        if not self.login_successful:
            return "Not logged in."
        try:
            greeks = options.get_option_market_data_by_id(option_id)
            return greeks if greeks else {}
        except Exception as e:
            print(f"Error fetching Greeks for option ID {option_id}: {e}")
            return {}

    def export_stock_orders(self, dir_path, file_name=None):
        """
        Export all stock orders to a CSV file.

        :param dir_path: Absolute or relative path to the directory the file will be written.
        :type dir_path: str
        :param file_name: An optional argument for the name of the file. If not defined, filename will be stock_orders_{current date}
        :type file_name: Optional[str]
        """
        r.export_completed_stock_orders(dir_path, file_name=file_name)

    def export_option_orders(self, dir_path, file_name=None):
        """
        Export all option orders to a CSV file.

        :param dir_path: Absolute or relative path to the directory the file will be written.
        :type dir_path: str
        :param file_name: An optional argument for the name of the file. If not defined, filename will be option_orders_{current date}
        :type file_name: Optional[str]
        """
        r.export_completed_option_orders(dir_path, file_name=file_name)

    # def downloadPdf(
    #     self,
    #     doc_type="account_statement",
    #     start_date="2000-01-01",
    #     end_date=str(datetime.now().date()),
    # ):
    #     """
    #     Downloads documents like account statements or trade confirmations.

    #     :param doc_type: 'account_statement', 'tax_document', or 'trade_confirmation'.
    #     """
    #     if not self.login_successful:
    #         return "Not logged in."
    #     print(
    #         f"\nDownloading '{doc_type}' documents from {start_date} to {end_date}..."
    #     )
    #     docs = r.get_documents(doctype=doc_type)

    #     output_dir = path.Path("RH_pdf")
    #     output_dir.mkdir(exist_ok=True)

    #     start = datetime.strptime(start_date, "%Y-%m-%d").date()
    #     end = datetime.strptime(end_date, "%Y-%m-%d").date()

    #     download_count = 0
    #     for doc in docs:
    #         doc_date = datetime.strptime(doc["date"], "%Y-%m-%d").date()
    #         if start <= doc_date <= end:
    #             doc_id = doc["id"]
    #             print(f"  -> Downloading document ID: {doc_id}")
    #             r.download_document_by_id(
    #                 doc_id, dir_path=str(output_dir), file_name=f"{doc_id}.pdf"
    #             )
    #             download_count += 1

    #     print(f"\nDownloaded {download_count} documents to the '{output_dir}' folder.")


# --- Main execution block for testing ---
if __name__ == "__main__":

    # --- Step 1: Create an instance of the class to log in ---
    rh = Robinhood()

    # --- Step 2: Use the class methods if login was successful ---
    if rh.login_successful:

        # --- Example Usage ---
        # You can uncomment the lines below to test the functions

        # # test get_all_stock_orders
        # orders = rh.get_all_stock_orders()
        # print("\n--- All stock orders ---")
        # print(orders.columns)

        # # test get_all_stock_positions
        # all_stock_positions = rh.get_all_stock_positions()
        # print("\n--- All Stock Positions ---")
        # print(all_stock_positions.columns)

        # test get_instrumentBasic
        instrument_data = rh.get_instrumentBasic(
            "https://api.robinhood.com/instruments/450dfc6d-5510-4d40-abfb-f633b7d9be3e/"
        )
        print("\n--- Instrument Data ---")
        print(instrument_data)

        # # Get open option positions
        # open_options = rh.get_open_options_positions()
        # print("\n--- Open Option Positions ---")
        # print(open_options.columns)

        # 4. Download recent trade confirmations
        # rh.downloadPdf(doc_type='trade_confirmation', start_date='2025-07-01')

        # test export_stock_orders
        rh.export_stock_orders("./data")
        print("\n--- Stock orders exported successfully ---")

        # test export_option_orders
        rh.export_option_orders("./data")
        print("\n--- Option orders exported successfully ---")

        # --- Step 3: Log out when you are finished ---
        rh._logout()
