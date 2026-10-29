#!/usr/bin/env python3
"""Minimal BRef MiLB test - just AAA with short date range."""
import sys
sys.path.insert(0, '/opt/anaconda3/lib/python3.12/site-packages')
import os
os.environ['PYBASEBALL_CACHE'] = '/tmp/pybaseball_cache'

from pybaseball.datasources.bref import BRefSession
from bs4 import BeautifulSoup
import pandas as pd
import json

session = BRefSession()

# Try smallest possible request
url = "http://www.baseball-reference.com/leagues/daily.cgi?type=b&level=AAA&lastndays=7"
print(f"Fetching: {url}")
resp = session.get(url)
print(f"Status: {resp.status_code}, Size: {len(resp.content)} bytes")

soup = BeautifulSoup(resp.content, 'lxml')
tables = soup.find_all('table')
print(f"Tables found: {len(tables)}")

if tables:
    table = tables[0]
    headers = [th.get_text().strip() for th in table.find("tr").find_all("th")]
    print(f"Headers: {headers}")
    
    rows = table.find('tbody').find_all('tr')
    data_rows = [r for r in rows if r.find_all('td')]
    print(f"Data rows: {len(data_rows)}")
    
    # Get first 5 rows with mlbID
    for i, row in enumerate(data_rows[:5]):
        cols = [ele.text.strip() for ele in row.find_all('td')]
        row_anchor = row.find("a")
        mlbid = None
        if row_anchor and "mlb_ID=" in row_anchor.get("href", ""):
            href = row_anchor["href"]
            mlbid = href.split("mlb_ID=")[-1].split("&")[0] if "mlb_ID=" in href else None
        print(f"  Row {i}: cols={cols[:8]}, mlbID={mlbid}")

# Now try AA
print("\n--- AA ---")
url2 = "http://www.baseball-reference.com/leagues/daily.cgi?type=b&level=AA&lastndays=7"
resp2 = session.get(url2)
soup2 = BeautifulSoup(resp2.content, 'lxml')
tables2 = soup2.find_all('table')
if tables2:
    rows2 = tables2[0].find('tbody').find_all('tr')
    data_rows2 = [r for r in rows2 if r.find_all('td')]
    print(f"AA data rows: {len(data_rows2)}")

# Now try A+
print("\n--- A+ ---")
url3 = "http://www.baseball-reference.com/leagues/daily.cgi?type=b&level=A%2B&lastndays=7"
resp3 = session.get(url3)
soup3 = BeautifulSoup(resp3.content, 'lxml')
tables3 = soup3.find_all('table')
if tables3:
    rows3 = tables3[0].find('tbody').find_all('tr')
    data_rows3 = [r for r in rows3 if r.find_all('td')]
    print(f"A+ data rows: {len(data_rows3)}")

print("\nDone!")
