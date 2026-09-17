"""验证 /api/model_type_config 新增的基座信息字段"""
import json, urllib.request

for t in ["minimind", "deepseek-r1-1.5b"]:
    r = json.loads(urllib.request.urlopen(
        f"http://127.0.0.1:7861/api/model_type_config?type={t}", timeout=5).read())
    print(f"=== {t} ===")
    print(f"  display_name:        {r.get('display_name')}")
    print(f"  base_model_rel:      {r.get('base_model_rel')}")
    print(f"  base_model_size_mb:  {r.get('base_model_size_mb')}")
    print(f"  base_available:      {r.get('base_available')}")
