import json

index = json.load(open("web/public/data/players_index.json"))

for p in index:
    if p.get("name") == "Jake Brown" and p.get("school_abb") == "LSU":
        pid = p["id"]
        print(f"Found LSU Jake Brown: pid={pid}")
        
        h = 0x811c9dc5
        for b in pid.encode('utf-8'):
            h ^= b
            h = (h * 0x01000193) & 0xFFFFFFFF
        shard_num = h % 64
        shard_path = f"web/public/data/players/shard-{shard_num:02d}.json"
        print(f"Shard: {shard_path}")
        
        shard = json.load(open(shard_path))
        if pid in shard:
            detail = shard[pid]
            seasons = detail.get("seasons", [])
            print(f"Seasons: {len(seasons)}")
            for s in seasons:
                team_abb = s.get("team_abb", "-")
                wOBA = s.get("wOBA", "-")
                print(f"  {s.get('Season')}: team={team_abb} wOBA={wOBA} PA={s.get('PA','-')}")
        break
