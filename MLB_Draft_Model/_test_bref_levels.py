#!/usr/bin/env python3
"""Test BRef MiLB endpoints for all levels and check coverage."""
from pybaseball.datasources.bref import BRefSession
from bs4 import BeautifulSoup
import pandas as pd
import json, re, sys, time

session = BRefSession()

# Test multiple MiLB levels
levels = ['mlb', 'AAA', 'AA', 'A+', 'A', 'A-', 'Rk', 'Fg']
# Also try numeric codes
level_codes = range(1, 12)  # Different possible level codes

print("=" * 60)
print("TEST: Different MiLB levels on BRef daily.cgi")
print("=" * 60)

for level in levels:
    url = f"http://www.baseball-reference.com/leagues/daily.cgi?user_team=&bust_cache=&type=b&lastndays=365&dates=fromandto&fromandto=2024-04-01.2024-10-01&level={level}&franch=&stat=&stat_value=0"
    try:
        resp = session.get(url)
        soup = BeautifulSoup(resp.content, 'lxml')
        tables = soup.find_all('table')
        if tables:
            rows = tables[0].find_all('tr')
            data_rows = [r for r in rows if r.find_all('td')]
            print(f"  {level}: {len(data_rows)} players, status={resp.status_code}")
        else:
            print(f"  {level}: No table, status={resp.status_code}")
    except Exception as e:
        print(f"  {level}: ERROR - {e}")
    time.sleep(0.5)  # Rate limiting

print("\n" + "=" * 60)
print("TEST: Try numeric level codes")
print("=" * 60)
for code in level_codes:
    url = f"http://www.baseball-reference.com/leagues/daily.cgi?type=b&level={code}&lastndays=365&dates=fromandto&fromandto=2024-04-01.2024-10-01"
    try:
        resp = session.get(url)
        soup = BeautifulSoup(resp.content, 'lxml')
        tables = soup.find_all('table')
        if tables:
            rows = tables[0].find_all('tr')
            data_rows = [r for r in rows if r.find_all('td')]
            if data_rows:
                print(f"  code={code}: {len(data_rows)} players, status={resp.status_code}")
        else:
            pass  # Skip levels with no data
    except Exception as e:
        pass
    time.sleep(0.3)

print("\n" + "=" * 60)
print("TEST: Get AAA batting data with mlbID crosswalk")
print("=" * 60)

# Fetch AAA batting data for 2024
url = "http://www.baseball-reference.com/leagues/daily.cgi?user_team=&bust_cache=&type=b&lastndays=365&dates=fromandto&fromandto=2024-04-01.2024-10-01&level=AAA&franch=&stat=&stat_value=0"
resp = session.get(url)
soup = BeautifulSoup(resp.content, 'lxml')
table = soup.find_all('table')[0]

# Extract headers
headers = []
for th in table.find("tr").find_all("th"):
    headers.append(th.get_text().strip())
# The first header is usually 'Rk' which we skip
headers = headers[1:]
headers.append('mlbID')
print(f"Columns: {headers}")

# Extract data rows
data = []
for row in table.find('tbody').find_all('tr'):
    cols = row.find_all('td')
    if not cols:
        continue
    row_anchor = row.find("a")
    mlbid = row_anchor["href"].split("mlb_ID=")[-1] if row_anchor and "mlb_ID=" in (row_anchor.get("href", "")) else None
    col_data = [ele.text.strip() for ele in cols]
    # First col might be 'Rk'
    col_data = col_data[1:] if len(col_data) > len(headers) - 1 else col_data
    col_data.append(mlbid)
    data.append(col_data)

df = pd.DataFrame(data)
if len(df.columns) == len(headers):
    df.columns = headers
else:
    print(f"Column mismatch: {len(df.columns)} data cols vs {len(headers)} headers")
    print(f"First row has {len(data[0])} values")
    
print(f"\nAAA batting 2024: {len(df)} players")
print(f"Sample (first 3):")
print(df.head(3).to_string())

# Count how many have mlbID
with_mlbid = df['mlbID'].notna().sum() if 'mlbID' in df.columns else 0
print(f"\nPlayers with mlbID: {with_mlbid} / {len(df)}")

# Save to CSV for inspection
df.to_csv('/tmp/aaa_batting_2024.csv', index=False)
print(f"Saved to /tmp/aaa_batting_2024.csv")

print("\n" + "=" * 60)
print("TEST: Now get pitching for AAA 2024")
print("=" * 60)

url = "http://www.baseball-reference.com/leagues/daily.cgi?user_team=&bust_cache=&type=p&lastndays=365&dates=fromandto&fromandto=2024-04-01.2024-10-01&level=AAA&franch=&stat=&stat_value=0"
resp = session.get(url)
soup = BeautifulSoup(resp.content, 'lxml')
table = soup.find_all('table')[0]

headers = []
for th in table.find("tr").find_all("th"):
    headers.append(th.get_text().strip())
headers = headers[1:]
headers.append('mlbID')

data = []
for row in table.find('tbody').find_all('tr'):
    cols = row.find_all('td')
    if not cols:
        continue
    row_anchor = row.find("a")
    mlbid = row_anchor["href"].split("mlb_ID=")[-1] if row_anchor and "mlb_ID=" in (row_anchor.get("href", "")) else None
    col_data = [ele.text.strip() for ele in cols]
    col_data = col_data[1:] if len(col_data) > len(headers) - 1 else col_data
    col_data.append(mlbid)
    data.append(col_data)

df_p = pd.DataFrame(data, columns=headers)
print(f"AAA pitching 2024: {len(df_p)} players")
print(f"Sample (first 2):")
print(df_p.head(2).to_string())

with_mlbid = df_p['mlbID'].notna().sum() if 'mlbID' in df_p.columns else 0
print(f"Players with mlbID: {with_mlbid} / {len(df_p)}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
