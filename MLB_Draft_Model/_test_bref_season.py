#!/usr/bin/env python3
"""Test full-season MiLB data with date range."""
import sys, os
os.environ['PYBASEBALL_CACHE'] = '/tmp/pybaseball_cache'
sys.path.insert(0, '/opt/anaconda3/lib/python3.12/site-packages')

from pybaseball.datasources.bref import BRefSession
from bs4 import BeautifulSoup
import pandas as pd
import json

session = BRefSession()

# Try a single month first
url = "http://www.baseball-reference.com/leagues/daily.cgi?type=b&level=AAA&lastndays=365&dates=fromandto&fromandto=2024-06-01.2024-06-30"
print(f"Fetching AAA batting Jun 2024...")
resp = session.get(url)
print(f"Status: {resp.status_code}, Size: {len(resp.content)} bytes")

soup = BeautifulSoup(resp.content, 'lxml')
table = soup.find_all('table')[0]
headers = [th.get_text().strip() for th in table.find("tr").find_all("th")]
print(f"Headers ({len(headers)}): {headers}")

rows = table.find('tbody').find_all('tr')
data_rows = [r for r in rows if r.find_all('td')]
print(f"Data rows: {len(data_rows)}")

# Parse into a proper dataframe with mlbID
parsed = []
for row in data_rows:
    cols = [ele.text.strip() for ele in row.find_all('td')]
    row_anchor = row.find("a")
    mlbid = None
    if row_anchor and "mlb_ID=" in row_anchor.get("href", ""):
        href = row_anchor["href"]
        mlbid = href.split("mlb_ID=")[-1].split("&")[0] if "mlb_ID=" in href else None
    # Skip first 'Rk' column if present
    start_idx = 1 if cols and cols[0].isdigit() else 0
    col_data = cols[start_idx:start_idx + len(headers) - 1]
    col_data.append(mlbid)
    parsed.append(col_data)

if parsed:
    df = pd.DataFrame(parsed)
    if df.shape[1] == len(headers):
        df.columns = headers
    print(f"Parsed DataFrame: {df.shape}")
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())
    
    # Count mlbID presence
    has_id = df['mlbID'].notna().sum() if 'mlbID' in df.columns else 0
    print(f"\nRows with mlbID: {has_id} / {len(df)}")
    
    # Level distribution
    if 'Lev' in df.columns:
        print(f"\nLevel distribution:")
        print(df.groupby('Lev').size())

# Now try the full season (Apr-Oct) with caching
print("\n--- Full season AAA batting 2024 ---")
url2 = "http://www.baseball-reference.com/leagues/daily.cgi?type=b&level=AAA&lastndays=365&dates=fromandto&fromandto=2024-04-01.2024-10-01"
resp2 = session.get(url2)
print(f"Status: {resp2.status_code}, Size: {len(resp2.content)} bytes")
soup2 = BeautifulSoup(resp2.content, 'lxml')
table2 = soup2.find_all('table')[0]
rows2 = table2.find('tbody').find_all('tr')
data_rows2 = [r for r in rows2 if r.find_all('td')]
print(f"Full season data rows: {len(data_rows2)}")

# Save the raw parsed data for inspection
parsed2 = []
for row in data_rows2:
    cols = [ele.text.strip() for ele in row.find_all('td')]
    row_anchor = row.find("a")
    mlbid = None
    if row_anchor and "mlb_ID=" in row_anchor.get("href", ""):
        href = row_anchor["href"]
        mlbid = href.split("mlb_ID=")[-1].split("&")[0] if "mlb_ID=" in href else None
    start_idx = 1 if cols and cols[0].isdigit() else 0
    col_data = cols[start_idx:start_idx + len(headers) - 1]
    col_data.append(mlbid)
    parsed2.append(col_data)

if parsed2:
    df2 = pd.DataFrame(parsed2)
    if df2.shape[1] == len(headers):
        df2.columns = headers
    print(f"Full season parsed: {df2.shape}")
    
    # Level distribution
    if 'Lev' in df2.columns:
        print(f"\nLevel distribution:")
        print(df2.groupby('Lev').size())

    # Save
    df2.to_csv('/tmp/aaa_batting_2024.csv', index=False)
    print(f"\nSaved to /tmp/aaa_batting_2024.csv")
    
    # Check for players with no mlbID
    no_id = df2[df2['mlbID'].isna()] if 'mlbID' in df2.columns else None
    if no_id is not None:
        print(f"\nRows without mlbID: {len(no_id)}")

print("\nDone!")
