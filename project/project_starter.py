import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# Locate the CSV files relative to this script, so the project runs from any folder
# (and on a reviewer's machine) without hard-coded absolute paths.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def data_path(filename: str) -> str:
    """Return the path of a data file: next to this script, in ./project, or the CWD."""
    for folder in (SCRIPT_DIR, os.path.join(SCRIPT_DIR, "project"), os.getcwd()):
        candidate = os.path.join(folder, filename)
        if os.path.exists(candidate):
            return candidate
    return filename

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                    - item_name
                    - category
                    - unit_price
                    - current_stock
                    - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                            Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv(data_path("quote_requests.csv"))
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv(data_path("quotes.csv"))
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# =====================================================================
# Multi-agent system (smolagents)
#
# Architecture — 4 agents total (limit is 5):
#   1. Orchestrator agent : reads the customer request, delegates to workers,
#                           writes the final customer-facing reply.
#   2. Inventory agent    : stock checks, catalog matching, restocking.
#   3. Quoting agent      : quote history lookup + deterministic pricing
#                           with bulk discounts.
#   4. Sales agent        : finalizes orders (re-checks stock, reorders from
#                           the supplier when needed and possible, records
#                           the sale) and reports on finances.
#
# Design principles:
#   * All arithmetic, stock math and database writes live in tools, so the
#     LLM never invents prices or quantities.
#   * Each worker agent only gets the tools it needs.
#   * The orchestrator delegates through explicit tools that call the worker
#     agents, which keeps the delegation visible and works across
#     smolagents versions.
# =====================================================================

import difflib
import inspect
import re
import threading

from smolagents import OpenAIServerModel, ToolCallingAgent, tool

# ---------------------------------------------------------------------
# Environment and model
# ---------------------------------------------------------------------
dotenv.load_dotenv()

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY") or os.getenv("UDACITY_OPENAI_API_KEY"),
)

# ---------------------------------------------------------------------
# Business rules (single source of truth, used by several tools)
# ---------------------------------------------------------------------
CATALOG: Dict[str, Dict] = {item["item_name"]: item for item in paper_supplies}

# Bulk discount tiers, keyed on the total number of units in the order.
# Ordered from largest threshold to smallest.
DISCOUNT_TIERS = [
    (10000, 0.15),
    (5000, 0.10),
    (1000, 0.05),
    (0, 0.00),
]

DEFAULT_MIN_STOCK = 100      # reorder buffer for items not in the inventory table
CASH_SAFETY_RESERVE = 1000.0  # never spend cash below this level on restocking

# Per-request memory, reset by call_multi_agent_system before each request.
# - request_date / needed_by: parsed from the request so tools never use made-up dates
# - fulfilled / delegations: stop agents repeating work (duplicate sales, re-delegation)
# The lock matters because ToolCallingAgent can run several tool calls in parallel.
REQUEST_STATE: Dict = {"fulfilled": {}, "delegations": {}, "request_date": None, "needed_by": None}
STATE_LOCK = threading.Lock()
FULFILL_LOCK = threading.Lock()


def _reset_request_state(request_date: Union[str, None] = None, needed_by: Union[str, None] = None) -> None:
    REQUEST_STATE.update(fulfilled={}, delegations={}, request_date=request_date, needed_by=needed_by)

# Common customer phrasings that difflib alone matches poorly.
ALIASES = {
    "a4": "A4 paper",
    "printer paper": "Standard copy paper",
    "printing paper": "Standard copy paper",
    "copy paper": "Standard copy paper",
    "office paper": "Standard copy paper",
    "letter paper": "Letter-sized paper",
    "letter-size paper": "Letter-sized paper",
    "letter size paper": "Letter-sized paper",
    "legal paper": "Legal-size paper",
    "card stock": "Cardstock",
    "heavy cardstock": "Cardstock",
    "colorful paper": "Colored paper",
    "construction": "Construction paper",
    "poster board": "Poster paper",
    "posters": "Large poster paper (24x36 inches)",
    "large posters": "Large poster paper (24x36 inches)",
    "large poster paper": "Large poster paper (24x36 inches)",
    "large poster": "Large poster paper (24x36 inches)",
    "banner rolls": "Rolls of banner paper (36-inch width)",
    "banner": "Rolls of banner paper (36-inch width)",
    "banners": "Rolls of banner paper (36-inch width)",
    "washi tape": "Decorative adhesive tape (washi tape)",
    "napkins": "Paper napkins",
    "plates": "Paper plates",
    "cups": "Paper cups",
    "streamers": "Party streamers",
    "party bags": "Paper party bags",
    "name tags": "Name tags with lanyards",
    "folders": "Presentation folders",
    "tablecloths": "Table covers",
    "table cloths": "Table covers",
    "invitations": "Invitation cards",
    "cover stock": "100 lb cover stock",
    "text paper": "80 lb text paper",
}


def _date_only(date_str: str) -> str:
    """Normalize any date/datetime string to YYYY-MM-DD."""
    return str(date_str).strip().split("T")[0].split(" ")[0]


def _request_date(llm_date: str = "") -> str:
    """
    The date every tool works on. The model sometimes invents dates (e.g. 2023),
    which would hide all stock, so the request date parsed in
    call_multi_agent_system always wins over what the model passes.
    """
    return REQUEST_STATE.get("request_date") or _date_only(llm_date)


def _needed_by_date(llm_date: str = "") -> str:
    """Customer deadline: parsed from the request text, else the model's value if sane."""
    request_date = _request_date(llm_date)
    if REQUEST_STATE.get("needed_by"):
        return REQUEST_STATE["needed_by"]
    candidate = _date_only(llm_date)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", candidate) and candidate >= request_date:
        return candidate
    return request_date


MONTHS = ("january|february|march|april|may|june|july|august|"
        "september|october|november|december")


def _parse_needed_by(text: str, request_date: str) -> Union[str, None]:
    """Find the delivery deadline in the request text, e.g. 'by April 15, 2025'."""
    found = []
    for month, day, year in re.findall(rf"({MONTHS})\s+(\d{{1,2}}),?\s+(\d{{4}})", text, re.I):
        try:
            found.append(datetime.strptime(f"{month} {day} {year}", "%B %d %Y").strftime("%Y-%m-%d"))
        except ValueError:
            pass
    found += re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    later = sorted(d for d in found if d >= request_date)
    return later[-1] if later else None


# Words that describe size, quantity or quality rather than the product type.
GENERIC_WORDS = {
    "paper", "papers", "sheets", "sheet", "of", "the", "and", "with", "a", "for",
    "size", "sized", "high", "quality", "various", "assorted", "colors", "colours",
    "rolls", "roll", "reams", "ream", "packs", "pack", "units", "pieces", "white",
    "standard", "x", "in", "inch", "inches",
}


def _clean_item_text(text: str) -> str:
    """Drop leading quantities and units: '10,000 sheets of A4 paper' -> 'a4 paper'."""
    text = text.strip().lower()
    text = re.sub(r"^[\d,\.\s]+", "", text)
    text = re.sub(r"^(sheets|sheet|reams|ream|rolls|roll|packs|pack|units|pieces|boxes|box)\s+(of\s+)?", "", text)
    return text.strip(" .")


def _match_catalog(requested_name: str) -> Union[str, None]:
    """Return the closest catalog item name, or None when nothing is close."""
    name = _clean_item_text(requested_name)
    if not name:
        return None
    lower_map = {n.lower(): n for n in CATALOG}
    if name in lower_map:
        return lower_map[name]
    if name in ALIASES:
        return ALIASES[name]
    # Multi-word aliases are specific, so check them before fuzzy matching
    # ("decorative washi tape" -> washi tape, not Decorative paper).
    for alias in sorted((a for a in ALIASES if " " in a), key=len, reverse=True):
        if alias in name:
            return ALIASES[alias]
    # Near-identical spelling only ("glossy papers"); a high cutoff keeps
    # "A3 paper" from being sold as "A4 paper".
    close = difflib.get_close_matches(name, lower_map.keys(), n=1, cutoff=0.88)
    if close:
        return lower_map[close[0]]
    # Otherwise prefer the catalog item whose distinctive words appear in the request
    # ("A4 glossy paper" -> "Glossy paper", "heavy cardstock" -> "Cardstock").
    words = set(re.findall(r"[a-z0-9]+", name)) - GENERIC_WORDS
    if len(words) > 1:
        words.discard("a4")  # "A4" is a size; another word names the paper type
    best, best_key = None, (0, 0.0)
    for lower_name, catalog_name in lower_map.items():
        key_words = set(re.findall(r"[a-z0-9]+", lower_name)) - GENERIC_WORDS
        if key_words:
            overlap = len(words & key_words)
            key = (overlap, overlap / len(key_words))  # most shared words, then best coverage
            if key > best_key:
                best, best_key = catalog_name, key
    if best_key[1] >= 0.5:
        return best
    for alias, catalog_name in ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", name):
            return catalog_name
    return None


def _current_stock(item_name: str, as_of_date: str) -> int:
    df = get_stock_level(item_name, as_of_date)
    value = df["current_stock"].iloc[0] if not df.empty else 0
    return int(value) if pd.notna(value) else 0


def _min_stock_level(item_name: str) -> int:
    inv = pd.read_sql(
        "SELECT min_stock_level FROM inventory WHERE item_name = :name",
        db_engine,
        params={"name": item_name},
    )
    return int(inv["min_stock_level"].iloc[0]) if not inv.empty else DEFAULT_MIN_STOCK


def _discount_rate(total_units: int) -> float:
    for threshold, rate in DISCOUNT_TIERS:
        if total_units >= threshold:
            return rate
    return 0.0


def _place_stock_order(item_name: str, quantity: int, order_date: str) -> Dict:
    """Buy stock from the supplier if cash allows. Shared by inventory and sales tools."""
    unit_price = CATALOG[item_name]["unit_price"]
    cost = round(quantity * unit_price, 2)
    cash = get_cash_balance(order_date)
    if cash - cost < CASH_SAFETY_RESERVE:
        return {"ok": False, "reason": f"insufficient cash (balance ${cash:.2f}, cost ${cost:.2f})"}
    delivery = get_supplier_delivery_date(order_date, quantity)
    txn_id = create_transaction(item_name, "stock_orders", quantity, cost, order_date)
    return {"ok": True, "transaction_id": txn_id, "cost": cost, "delivery_date": delivery}


# ---------------------------------------------------------------------
# Tools for inventory agent
# ---------------------------------------------------------------------
@tool
def find_catalog_items_tool(requested_items: str) -> str:
    """
    Map the item names a customer used to the exact item names in the company catalog.
    Always call this before checking stock, quoting or selling.

    Args:
        requested_items: Item names separated by semicolons, as the customer wrote
            them, e.g. "A4 glossy paper; heavy cardstock; balloons".

    Returns:
        One line per requested item with the matched catalog name and unit price,
        or NOT SOLD when the company does not carry the item.
    """
    lines = []
    for raw in re.split(r";|,(?!\d{3}\b)", requested_items):
        raw = raw.strip()
        if not raw:
            continue
        match = _match_catalog(raw)
        if match:
            lines.append(f"'{raw}' -> '{match}' (unit price ${CATALOG[match]['unit_price']:.2f})")
        else:
            lines.append(f"'{raw}' -> NOT SOLD (no matching catalog item)")
    return "\n".join(lines) if lines else "No items provided."


@tool
def check_inventory_tool(as_of_date: str) -> str:
    """
    List every item currently in stock with its quantity as of a date.

    Args:
        as_of_date: Date in YYYY-MM-DD format.

    Returns:
        A text table of item names and units in stock.
    """
    inventory = get_all_inventory(_request_date(as_of_date))
    if not inventory:
        return "No items in stock."
    return "\n".join(f"{name}: {int(qty)} units" for name, qty in sorted(inventory.items()))


@tool
def check_stock_level_tool(item_name: str, as_of_date: str) -> str:
    """
    Get the stock level of one catalog item, plus its reorder threshold.

    Args:
        item_name: Exact catalog item name (use find_catalog_items_tool first).
        as_of_date: Date in YYYY-MM-DD format.

    Returns:
        Current stock, minimum stock level and whether the item needs restocking.
    """
    if item_name not in CATALOG:
        return f"'{item_name}' is not a catalog item. Use find_catalog_items_tool first."
    stock = _current_stock(item_name, _request_date(as_of_date))
    min_level = _min_stock_level(item_name)
    status = "BELOW minimum - restock recommended" if stock < min_level else "OK"
    return f"{item_name}: {stock} units in stock (minimum level {min_level}) -> {status}"


@tool
def get_delivery_date_tool(order_date: str, quantity: int) -> str:
    """
    Estimate when a supplier order placed on order_date would arrive.

    Args:
        order_date: Date the supplier order is placed, YYYY-MM-DD.
        quantity: Number of units to order.

    Returns:
        The estimated supplier delivery date.
    """
    order_date = _request_date(order_date)
    return f"Supplier delivery for {quantity} units ordered {order_date}: {get_supplier_delivery_date(order_date, int(quantity))}"


@tool
def check_cash_balance_tool(as_of_date: str) -> str:
    """
    Get the company's cash balance on a date.

    Args:
        as_of_date: Date in YYYY-MM-DD format.

    Returns:
        The cash balance in dollars.
    """
    as_of_date = _request_date(as_of_date)
    return f"Cash balance as of {as_of_date}: ${get_cash_balance(as_of_date):.2f}"


@tool
def reorder_stock_tool(item_name: str, quantity: int, order_date: str) -> str:
    """
    Place a stock order with the supplier. Refuses if the purchase would push cash
    below the safety reserve.

    Args:
        item_name: Exact catalog item name.
        quantity: Number of units to buy.
        order_date: Date the order is placed, YYYY-MM-DD.

    Returns:
        Confirmation with cost and supplier delivery date, or the reason it was refused.
    """
    if item_name not in CATALOG:
        return f"Cannot reorder '{item_name}': not a catalog item."
    if int(quantity) <= 0:
        return "Quantity must be positive."
    result = _place_stock_order(item_name, int(quantity), _request_date(order_date))
    if not result["ok"]:
        return f"Reorder of {item_name} refused: {result['reason']}."
    return (f"Ordered {quantity} units of {item_name} for ${result['cost']:.2f}; "
            f"supplier delivery {result['delivery_date']} (transaction {result['transaction_id']}).")


# ---------------------------------------------------------------------
# Tools for quoting agent
# ---------------------------------------------------------------------
@tool
def search_quote_history_tool(search_terms: str) -> str:
    """
    Find similar past quotes to keep pricing consistent.

    Args:
        search_terms: Comma-separated keywords, e.g. "cardstock, wedding". Use 1-3
            short terms; every term must match, so fewer terms find more quotes.

    Returns:
        Up to 5 past quotes with amount, order size, event type and explanation.
    """
    terms = [t.strip() for t in search_terms.split(",") if t.strip()]
    results = search_quote_history(terms, limit=5)
    if not results and len(terms) > 1:
        results = search_quote_history(terms[:1], limit=5)
    if not results:
        return "No similar past quotes found."
    lines = []
    for q in results:
        explanation = str(q.get("quote_explanation", ""))[:300]
        lines.append(f"- ${q['total_amount']} | {q['order_size']} order | {q['event_type']} | {explanation}")
    return "\n".join(lines)


@tool
def calculate_quote_tool(items: str) -> str:
    """
    Compute an exact quote with bulk discounts. Use this for every quote; never
    calculate prices by hand.

    Discount tiers by total units in the order: 1,000+ units 5%, 5,000+ units 10%,
    10,000+ units 15%.

    Args:
        items: Semicolon-separated "catalog item name:quantity" pairs, e.g.
            "A4 paper:500;Cardstock:200".

    Returns:
        Itemized prices, discount rate, discount amount and the final total.
    """
    parsed = []
    errors = []
    for chunk in items.split(";"):
        if not chunk.strip():
            continue
        if ":" not in chunk:
            errors.append(f"Could not parse '{chunk.strip()}'")
            continue
        name, qty = chunk.rsplit(":", 1)
        catalog_name = _match_catalog(name)
        digits = re.sub(r"[^\d]", "", qty)
        if not catalog_name or not digits:
            errors.append(f"Could not price '{chunk.strip()}'")
            continue
        parsed.append((catalog_name, int(digits)))

    if not parsed:
        return "No priceable items. " + " ".join(errors)

    total_units = sum(q for _, q in parsed)
    rate = _discount_rate(total_units)
    lines, subtotal = [], 0.0
    for name, qty in parsed:
        unit = CATALOG[name]["unit_price"]
        line_total = round(unit * qty, 2)
        subtotal += line_total
        lines.append(f"{name}: {qty} x ${unit:.2f} = ${line_total:.2f}")
    discount = round(subtotal * rate, 2)
    total = round(subtotal - discount, 2)
    lines += [
        f"Total units: {total_units}",
        f"Subtotal: ${subtotal:.2f}",
        f"Bulk discount: {int(rate * 100)}% (-${discount:.2f})",
        f"FINAL TOTAL: ${total:.2f}",
        f"DISCOUNT_RATE: {rate}",
    ]
    if errors:
        lines.append("Not priced: " + "; ".join(errors))
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Tools for ordering (sales) agent
# ---------------------------------------------------------------------
@tool
def fulfill_order_tool(item_name: str, quantity: int, discount_rate: float,
                    order_date: str, required_by_date: str) -> str:
    """
    Finalize the sale of one line item. The tool re-checks stock; if stock is short it
    restocks from the supplier, but only when the supplier can deliver by
    required_by_date and cash allows. Then it records the sale at the catalog price
    minus the discount.

    Args:
        item_name: Exact catalog item name.
        quantity: Units the customer is buying.
        discount_rate: Discount from the quote as a decimal, e.g. 0.05 for 5%.
        order_date: Date of the customer request, YYYY-MM-DD.
        required_by_date: Date the customer needs delivery, YYYY-MM-DD. Use the
            order date if the customer gave none.

    Returns:
        SOLD with price and delivery date, or NOT FULFILLED with the reason.
    """
    if item_name not in CATALOG:
        return f"NOT FULFILLED: '{item_name}' is not a catalog item."
    # Loop guard: each item is finalized at most once per customer request.
    # (Own lock, because STATE_LOCK is held by the delegation that runs this agent.)
    with FULFILL_LOCK:
        if item_name in REQUEST_STATE["fulfilled"]:
            return (f"ALREADY PROCESSED for this request: {REQUEST_STATE['fulfilled'][item_name]} "
                    f"Do NOT call fulfill_order_tool for this item again.")
        result = _fulfill(item_name, quantity, discount_rate, order_date, required_by_date)
        REQUEST_STATE["fulfilled"][item_name] = result
        return result


def _fulfill(item_name: str, quantity: int, discount_rate: float,
            order_date: str, required_by_date: str) -> str:
    """Stock check, optional restock, and sale recording for one line item."""
    quantity = int(quantity)
    if quantity <= 0:
        return "NOT FULFILLED: quantity must be positive."
    rate = min(max(float(discount_rate), 0.0), 0.15)
    order_date = _request_date(order_date)
    required_by = _needed_by_date(required_by_date)

    stock = _current_stock(item_name, order_date)
    delivery_date = order_date
    restock_note = ""

    if stock < quantity:
        shortfall = quantity - stock
        reorder_qty = shortfall + _min_stock_level(item_name)
        supplier_date = get_supplier_delivery_date(order_date, reorder_qty)
        if supplier_date > required_by:
            return (f"NOT FULFILLED: {item_name} has {stock} units; restocking {reorder_qty} "
                    f"units would arrive {supplier_date}, after the required date {required_by}.")
        result = _place_stock_order(item_name, reorder_qty, order_date)
        if not result["ok"]:
            return f"NOT FULFILLED: {item_name} short by {shortfall} units and restock refused: {result['reason']}."
        delivery_date = supplier_date
        restock_note = f" (restocked {reorder_qty} units for ${result['cost']:.2f})"

    price = round(quantity * CATALOG[item_name]["unit_price"] * (1 - rate), 2)
    txn_id = create_transaction(item_name, "sales", quantity, price, order_date)
    return (f"SOLD: {quantity} x {item_name} for ${price:.2f} "
            f"({int(rate * 100)}% discount), delivery by {delivery_date}{restock_note}. "
            f"Transaction {txn_id}.")


@tool
def financial_report_tool(as_of_date: str) -> str:
    """
    Summarize the company's finances on a date.

    Args:
        as_of_date: Date in YYYY-MM-DD format.

    Returns:
        Cash, inventory value, total assets and items below their minimum stock level.
    """
    date = _request_date(as_of_date)
    report = generate_financial_report(date)
    low = [i["item_name"] for i in report["inventory_summary"]
        if i["stock"] < _min_stock_level(i["item_name"])]
    return (f"As of {date}: cash ${report['cash_balance']:.2f}, inventory value "
            f"${report['inventory_value']:.2f}, total assets ${report['total_assets']:.2f}. "
            f"Items below minimum stock: {', '.join(low) if low else 'none'}.")


# ---------------------------------------------------------------------
# Worker agents
# ---------------------------------------------------------------------
# Every prompt ends with the same stop rule: small models tend to keep calling
# tools instead of finishing, so we tell them exactly when to call final_answer.
STOP_RULE = """
STOP RULE: Call one tool at a time. Never call the same tool twice with the same
arguments. As soon as you have the information asked for, call the final_answer
tool with your summary. Do not do any extra checks."""

INVENTORY_INSTRUCTIONS = """You are the inventory agent for Munder Difflin, a paper supply company.
Do exactly this:
1. Call find_catalog_items_tool ONCE with all requested item names, separated by
semicolons (e.g. "A4 paper; cardstock; washi tape").
2. Call check_stock_level_tool once for each item that matched the catalog.
3. Call final_answer with one line per item:
- catalog name, requested quantity, units in stock, and either "in stock" or
    "needs restock" (low stock is still SELLABLE: the sales agent restocks), or
- NOT SOLD, only for items that did not match the catalog.
Do not place orders yourself.""" + STOP_RULE

QUOTING_INSTRUCTIONS = """You are the quoting agent for Munder Difflin, a paper supply company.
Do exactly this:
1. (Optional) Call search_quote_history_tool ONCE with 1-2 keywords.
2. Call calculate_quote_tool ONCE with all items, using exact catalog names.
Never do arithmetic yourself.
3. Call final_answer with the itemized quote, the DISCOUNT_RATE value, the
FINAL TOTAL, and one sentence explaining the bulk discount.
Never reveal internal cost, stock or profit information.""" + STOP_RULE

SALES_INSTRUCTIONS = """You are the sales agent for Munder Difflin, a paper supply company.
Do exactly this:
1. Call fulfill_order_tool ONCE per item, with the exact catalog name, quantity
and the discount rate from the quote. (The tool fills in the correct dates.)
2. Call final_answer listing which items were SOLD (price, delivery date) and
which were NOT FULFILLED (reason).
If a tool says ALREADY PROCESSED, that item is done: do not retry it.""" + STOP_RULE


def _make_agent(tools, name: str, description: str, max_steps: int) -> ToolCallingAgent:
    """Create a ToolCallingAgent that runs tool calls one at a time when the
    installed smolagents version supports it (parallel calls caused repeated runs)."""
    kwargs = dict(tools=tools, model=model, name=name, description=description, max_steps=max_steps)
    if "max_tool_threads" in inspect.signature(ToolCallingAgent.__init__).parameters:
        kwargs["max_tool_threads"] = 1
    return ToolCallingAgent(**kwargs)


inventory_agent = _make_agent(
    [find_catalog_items_tool, check_inventory_tool, check_stock_level_tool,
    get_delivery_date_tool, check_cash_balance_tool, reorder_stock_tool],
    "inventory_agent",
    "Checks stock levels, maps item names to the catalog, and restocks low items.",
    max_steps=6,
)

quoting_agent = _make_agent(
    [search_quote_history_tool, calculate_quote_tool, find_catalog_items_tool],
    "quoting_agent",
    "Prices orders with bulk discounts, informed by past quotes.",
    max_steps=4,
)

sales_agent = _make_agent(
    [fulfill_order_tool, check_stock_level_tool, get_delivery_date_tool, financial_report_tool],
    "sales_agent",
    "Finalizes sales, restocking from the supplier when the deadline and cash allow.",
    max_steps=6,
)


# ---------------------------------------------------------------------
# Orchestrator agent: delegation tools + the agent itself
# ---------------------------------------------------------------------
NEXT_STEP = {
    "inventory": "NEXT STEP: call delegate_to_quoting_agent with every item that matched the "
                "catalog, including items that need restock (or final_answer if all are NOT SOLD).",
    "quoting": "NEXT STEP: call delegate_to_sales_agent with these items and the DISCOUNT_RATE.",
    "sales": "NEXT STEP: call final_answer now with the reply to the customer. Do not call any other tool.",
}
PREREQUISITE = {"quoting": "inventory", "sales": "quoting"}


def _delegate(agent, key: str, instructions: str, task: str) -> str:
    """
    Run a worker agent at most once per request, in the order
    inventory -> quoting -> sales. The lock serializes parallel tool calls from the
    orchestrator, so a second call always sees the first one's stored result.
    """
    with STATE_LOCK:
        done = REQUEST_STATE["delegations"]
        if key in done:
            return (f"You already asked the {key} agent for this request. Its answer was:\n"
                    f"{done[key]}\n{NEXT_STEP[key]}")
        needed = PREREQUISITE.get(key)
        if needed and needed not in done:
            return f"Not yet: call delegate_to_{needed}_agent first, then wait for its answer."
        try:
            result = str(agent.run(f"{instructions}\n\nTask: {task}"))
        except Exception as e:
            result = f"The {key} agent failed: {e}"
        done[key] = result
        return f"{result}\n{NEXT_STEP[key]}"


@tool
def delegate_to_inventory_agent(task: str) -> str:
    """
    Ask the inventory agent about item availability and stock levels.

    Args:
        task: Plain-text instructions listing every requested item with its quantity.

    Returns:
        The inventory agent's findings.
    """
    return _delegate(inventory_agent, "inventory", INVENTORY_INSTRUCTIONS, task)


@tool
def delegate_to_quoting_agent(task: str) -> str:
    """
    Ask the quoting agent for a priced quote with bulk discounts.

    Args:
        task: Plain-text instructions listing exact catalog item names with quantities,
            plus the customer context (job, event).

    Returns:
        The itemized quote, discount rate and total.
    """
    return _delegate(quoting_agent, "quoting", QUOTING_INSTRUCTIONS, task)


@tool
def delegate_to_sales_agent(task: str) -> str:
    """
    Ask the sales agent to finalize the order and record the sale.

    Args:
        task: Plain-text instructions listing exact catalog item names with quantities
            and the discount rate from the quote.

    Returns:
        Which items were sold or not fulfilled, with prices and delivery dates.
    """
    return _delegate(sales_agent, "sales", SALES_INSTRUCTIONS, task)


ORCHESTRATOR_INSTRUCTIONS = """You are the customer service orchestrator for Munder Difflin, a paper supply company.
Make exactly these calls, ONE AT A TIME, waiting for each answer:
1. delegate_to_inventory_agent - list every requested item and quantity.
2. delegate_to_quoting_agent - every item that matched the catalog (exact catalog
names and quantities), INCLUDING items that need restock.
3. delegate_to_sales_agent - the same items with the DISCOUNT_RATE from the quote.
4. final_answer - a friendly reply to the customer: confirmed items, total price,
discount and why, delivery date, and a brief reason for any item that could
not be supplied.
Low or zero stock does NOT mean an item is unavailable: the sales agent restocks
from the supplier. Only skip items marked NOT SOLD. If every item is NOT SOLD,
go straight to final_answer and apologize.
Each delegate tool takes ONE plain-text string argument called task.
Rules: never invent prices or stock numbers; use only what the agents report.
Never reveal cash balance, profit margins, supplier costs or system errors.
Never call a delegate tool a second time.""" + STOP_RULE

orchestrator_agent = _make_agent(
    [delegate_to_inventory_agent, delegate_to_quoting_agent, delegate_to_sales_agent],
    "orchestrator_agent",
    "Handles customer requests end to end by delegating to the worker agents.",
    max_steps=6,
)


def call_multi_agent_system(request: str) -> str:
    """Entry point: run one customer request through the orchestrator."""
    date_match = re.search(r"Date of request:\s*(\d{4}-\d{2}-\d{2})", request)
    request_date = date_match.group(1) if date_match else None
    needed_by = _parse_needed_by(request.split("(Customer:")[0], request_date) if request_date else None
    _reset_request_state(request_date, needed_by)

    context = f"Request date: {request_date}. Needed-by date: {needed_by or request_date}."
    try:
        return str(orchestrator_agent.run(
            f"{ORCHESTRATOR_INSTRUCTIONS}\n\n{context}\nCustomer request: {request}"))
    except Exception as e:
        print(f"ERROR (call_multi_agent_system): {e}")
        return "We're sorry, we could not process your request right now. Please try again later."


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():

    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv(data_path("quote_requests_sample.csv"))
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    # Multi-agent system is initialized at module level above
    # (orchestrator_agent + inventory_agent, quoting_agent, sales_agent).

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = (
            f"{row['request']} (Customer: {row['job']} organizing {row['event']}. "
            f"Date of request: {request_date})"
        )

        response = call_multi_agent_system(request_with_date)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()