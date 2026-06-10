import requests
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import time
from pathlib import Path

# Database setup
DB_PATH = Path(__file__).parent.parent / "data" / "news_data.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def create_database():
    """Create SQLite database schema for news data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_symbol TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT UNIQUE NOT NULL,
            source TEXT,
            publish_date DATETIME NOT NULL,
            article_date DATETIME,
            language TEXT,
            fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(url)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scrape_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_symbol TEXT NOT NULL,
            date_queried DATE NOT NULL,
            articles_fetched INTEGER,
            status TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database created at {DB_PATH}")


def fetch_news_from_gdelt(stock_symbol, start_date, end_date):
    """
    Fetch news from GDELT API for a given stock symbol and date range
    
    Args:
        stock_symbol: Stock ticker symbol (e.g., 'RELIANCE')
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
    
    Returns:
        List of articles or empty list if error
    """
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={stock_symbol}"
        "&mode=artlist"
        "&maxrecords=250"
        "&format=json"
        f"&startdatetime={start_date}000000"
        f"&enddatetime={end_date}235959"
    )
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'articles' in data:
            return data['articles']
        return []
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news for {stock_symbol}: {e}")
        return []


def store_articles_in_db(stock_symbol, articles):
    """Store fetched articles in SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    inserted_count = 0
    
    for article in articles:
        try:
            # Parse the date from article
            article_date = article.get('sedate', None)
            if article_date:
                article_datetime = datetime.strptime(str(article_date), '%Y%m%d%H%M%S')
            else:
                article_datetime = None
            
            cursor.execute('''
                INSERT OR IGNORE INTO news_articles 
                (stock_symbol, title, url, source, publish_date, article_date, language)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                stock_symbol,
                article.get('title', '')[:500],
                article.get('url', ''),
                article.get('domain', '')[:100],
                datetime.now(),
                article_datetime,
                article.get('language', 'en')
            ))
            inserted_count += 1
        
        except Exception as e:
            print(f"Error inserting article for {stock_symbol}: {e}")
            continue
    
    conn.commit()
    conn.close()
    
    return inserted_count


def log_scrape_activity(stock_symbol, articles_count, status='success'):
    """Log scraping activity"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO scrape_log (stock_symbol, date_queried, articles_fetched, status)
        VALUES (?, ?, ?, ?)
    ''', (
        stock_symbol,
        datetime.now().date(),
        articles_count,
        status
    ))
    
    conn.commit()
    conn.close()


def scrape_news_for_period(stocks, start_year=1997, end_year=2026):
    """
    Scrape news for multiple stocks across a date range
    
    Args:
        stocks: List of stock symbols
        start_year: Starting year (1997)
        end_year: Ending year (2026)
    """
    create_database()
    
    # Split into yearly chunks to avoid overwhelming API
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    
    current_date = start_date
    
    for stock in stocks:
        print(f"\n{'='*60}")
        print(f"Fetching news for {stock} from {start_year} to {end_year}")
        print(f"{'='*60}")
        
        # Fetch by year
        year_start = start_date
        while year_start < end_date:
            year_end = year_start + timedelta(days=364)  # ~1 year
            
            if year_end > end_date:
                year_end = end_date
            
            start_str = year_start.strftime('%Y%m%d')
            end_str = year_end.strftime('%Y%m%d')
            
            print(f"  Fetching {stock}: {start_str} to {end_str}...", end=' ', flush=True)
            
            articles = fetch_news_from_gdelt(stock, start_str, end_str)
            
            if articles:
                inserted = store_articles_in_db(stock, articles)
                log_scrape_activity(stock, len(articles), 'success')
                print(f"✓ Got {len(articles)} articles, stored {inserted}")
            else:
                print(f"✗ No articles found")
                log_scrape_activity(stock, 0, 'no_results')
            
            # Rate limiting - be respectful to GDELT API
            time.sleep(1)
            
            year_start = year_end + timedelta(days=1)


def get_stats():
    """Get statistics from the database"""
    conn = sqlite3.connect(DB_PATH)
    
    # Total articles by stock
    df_by_stock = pd.read_sql_query('''
        SELECT stock_symbol, COUNT(*) as article_count
        FROM news_articles
        GROUP BY stock_symbol
        ORDER BY article_count DESC
    ''', conn)
    
    # Date range of articles
    df_date_range = pd.read_sql_query('''
        SELECT 
            stock_symbol,
            MIN(article_date) as earliest_date,
            MAX(article_date) as latest_date,
            COUNT(*) as total_articles
        FROM news_articles
        WHERE article_date IS NOT NULL
        GROUP BY stock_symbol
    ''', conn)
    
    conn.close()
    
    print("\n" + "="*70)
    print("ARTICLE COUNT BY STOCK")
    print("="*70)
    print(df_by_stock.to_string(index=False))
    
    print("\n" + "="*70)
    print("DATE RANGE BY STOCK")
    print("="*70)
    print(df_date_range.to_string(index=False))
    
    return df_by_stock, df_date_range


def query_articles(stock_symbol=None, limit=10):
    """Query articles from database"""
    conn = sqlite3.connect(DB_PATH)
    
    if stock_symbol:
        query = f'''
            SELECT stock_symbol, title, url, source, article_date
            FROM news_articles
            WHERE stock_symbol = '{stock_symbol}'
            ORDER BY article_date DESC
            LIMIT {limit}
        '''
    else:
        query = f'''
            SELECT stock_symbol, title, url, source, article_date
            FROM news_articles
            ORDER BY article_date DESC
            LIMIT {limit}
        '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df


if __name__ == "__main__":
    # List of Indian stocks to scrape
    stocks = [
        "RELIANCE",
        "TCS",
        "HDFCBANK",
        "ICICIBANK",
        "SBIN",
        "BAJFINANCE",
        "BHARTIARTL",
        "HINDUNILVR",
        "LT",
        "LICI"
    ]
    
    # Scrape news from 1997 to 2026
    print("Starting news scraping from GDELT...")
    scrape_news_for_period(stocks, start_year=1997, end_year=2026)
    
    # Get statistics
    get_stats()
    
    # Example queries
    print("\n" + "="*70)
    print("RECENT RELIANCE ARTICLES")
    print("="*70)
    df_reliance = query_articles('RELIANCE', limit=5)
    print(df_reliance.to_string(index=False))
