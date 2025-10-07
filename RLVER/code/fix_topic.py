#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法：
  python fix_topic.py \
    --in data/train_profile.jsonl \
    --out data/train_profile.fixed.jsonl
"""

import argparse, json, re, shutil, sys, pathlib

RULES = [
    # (正则或关键词列表, 赋值的topic)
    (["结婚","婚姻","未婚夫","未婚妻","男友","女友","伴侣"], "婚恋/是否结婚"),
    (["同事","上司","领导","团队","项目","绩效","同部门","职场"], "职场人际/团队协作冲突"),
    (["儿子","女儿","孩子","青少年","家长会","叛逆","青春期"], "亲子教育/青春期沟通"),
    (["借款","欠款","偿还","房款","房子","房产","债务","抵押"], "家庭财务/亲属借贷纠纷"),
    (["成绩","考试","测验","作业","班主任","学习","学校","老师"], "校园/学业态度与动机"),
    (["共情","理解我的感受","情感支持","被理解"], "情感支持/共情需求"),
]

def infer_topic(rec: dict) -> str:
    text = " ".join(str(rec.get(k,"")) for k in ("scene","task","player")).lower()
    # 粗暴中文→小写对英文有效，中文不区分大小写问题不大
    for keys, topic in RULES:
        for k in keys:
            if k in text:
                return topic
    return "未分类"

def is_blank(v) -> bool:
    return v is None or (isinstance(v,str) and v.strip()=="")  # 允许空字符串当空白

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="outp", required=True)
    ap.add_argument("--backup", action="store_true", help="为输入文件创建 .bak 备份（仅当 --out 指向同一路径时有用）")
    args = ap.parse_args()

    in_path  = pathlib.Path(args.inp)
    out_path = pathlib.Path(args.outp)

    if not in_path.exists():
        print(f"[ERR] 输入文件不存在：{in_path}", file=sys.stderr); sys.exit(1)

    fixed, total, already = 0, 0, 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line=line.strip()
            if not line: continue
            total += 1
            obj = json.loads(line)

            if "topic" not in obj or is_blank(obj.get("topic")):
                obj["topic"] = infer_topic(obj)
                fixed += 1
            else:
                already += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # 若就地覆盖，按需做备份
    if out_path.resolve() == in_path.resolve() and args.backup:
        shutil.copyfile(in_path, str(in_path)+".bak")

    print(f"[OK] 处理完成：总计 {total} 行，修复(新增/填充) topic {fixed} 行，原本已有 {already} 行。")
    print(f"[OUT] 写入：{out_path}")

if __name__ == "__main__":
    main()
