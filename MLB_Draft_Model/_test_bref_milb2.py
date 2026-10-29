#!/usr/bin/env python3
"""Try to find actual MiLB data on BRef."""
import sys, os
os.environ['PYBASEBALL_CACHE'] = '/tmp/pybaseball_cache'
sys.path.insert(0, '/opt/anaconda3/lib/python3.12/site-packages')

from pybaseball.datasources.bref import BRefSession
from bs4 import BeautifulSoup

session = BRefSession()

# Test 1: Compare level=mlb vs level=AAA
print("=" * 60)
print("TEST 1: Compare BRef daily.cgi level parameters")
print("=" * 60)

for level in ['mlb', 'AAA', 'AA']:
    # Same date range, same type
    url = f"http://www.baseball-reference.com/leagues/daily.cgi?type=b&level={level}&lastndays=30&dates=fromandto&fromandto=2024-06-01.2024-06-07"
    resp = session.get(url)
    soup = BeautifulSoup(resp.content, 'lxml')
    
    # Check if there's a "Minor League" or similar heading
    text_lower = resp.text.lower()
    has_minor = 'minor' in text_lower
    has_aaa_text = 'aaa' in text_lower or 'triple-a' in text_lower
    has_maj = 'maj' in text_lower
    
    # Get first data row to see Lev values
    tables = soup.find_all('table')
    if tables:
        rows = tables[0].find('tbody').find_all('tr')
        data_rows = [r for r in rows if r.find_all('td')]
        lev_values = set()
        for r in data_rows[:10]:
            cols = r.find_all('td')
            for i, c in enumerate(cols):
                if c.get_text().strip().startswith('Maj') or c.get_text().strip().startswith('AAA') or '-' in c.get_text().strip():
                    lev_values.add(c.get_text().strip())
                    break
        print(f"  level={level}: {len(data_rows)} rows, minor={has_minor}, maj={has_maj}")
        print(f"    Sample Lev values: {list(lev_values)[:5]}")
    else:
        print(f"  level={level}: no table")

# Test 2: Try BRef MiLB register pages
print("\n" + "=" * 60)
print("TEST 2: Try BRef MiLB register league pages")
print("=" * 60)

# BRef has minor league pages at /register/league.cgi?group=Minor&class=AAA
urls = [
    "https://www.baseball-reference.com/register/league.cgi?group=Minor&class=AAA",
    "https://www.baseball-reference.com/register/league.cgi?group=Minor&class=AA",
    "https://www.baseball-reference.com/register/league.cgi?group=Minor&class=A%2B",
]
for url in urls:
    try:
        resp = session.get(url)
        soup = BeautifulSoup(resp.content, 'lxml')
        tables = soup.find_all('table')
        print(f"  {url.split('class=')[-1]}: status={resp.status_code}, {len(resp.content)} bytes, tables={len(tables)}")
        if tables:
            rows = tables[0].find_all('tr')
            print(f"    First table rows: {len(rows)}")
            if rows:
                cells = rows[0].find_all(['th','td'])
                print(f"    Headers: {[c.get_text().strip() for c in cells][:10]}")
    except Exception as e:
        print(f"  {url}: ERROR - {e}")

# Test 3: Try Stathead or other BRef MiLB access patterns
print("\n" + "=" * 60)
print("TEST 3: Try BRef minor league player search")
print("=" * 60)

# Search for a known player who played in MiLB in 2024
# Let's try searching by player ID
url = "https://www.baseball-reference.com/players/gl.fcgi?id=edmanto01&t=b&year=2024"
try:
    resp = session.get(url)
    soup = BeautifulSoup(resp.content, 'lxml')
    # Check if it links to minor league pages
    for a in soup.find_all('a'):
        href = a.get('href', '')
        if 'minor' in href.lower() or 'register' in href.lower() or 'milb' in href.lower():
            print(f"  Found MiLB link: {href}")
            break
    else:
        print(f"  No MiLB links found on player page")
    # Check if minor league section exists
    has_minor_section = 'minor' in resp.text.lower()
    print(f"  Has minor league content: {has_minor_section}")
    print(f"  Page size: {len(resp.content)} bytes")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 4: Check what the BRef minor league ID system looks like
print("\n" + "=" * 60)
print("TEST 4: Try BRef MiLB team page")
print("=" * 60)

# BRef minor league team page (Reno Aces - AAA)
url = "https://www.baseball-reference.com/register/team.cgi?id=243fa4da"
try:
    resp = session.get(url)
    print(f"  Team page: {len(resp.content)} bytes")
    has_minor = 'minor' in resp.text.lower()
    print(f"  Has minor content: {has_minor}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\nDone!")
