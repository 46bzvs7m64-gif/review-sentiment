# -*- coding: utf-8 -*-
"""
商品评论情感分析  sentiment.py
零第三方依赖：词典法 + 否定词窗口反转 + 程度副词加权 + 评价维度归集。
输出：情感分布、好评关键词 TOP、差评集中维度 TOP、差评原文清单、Markdown + 可视化 HTML 报告。

用法：
  python sentiment.py
  python sentiment.py --input demo_comments.csv --product 某跑鞋
真实使用时把电商后台/平台导出的评论CSV替换 demo_comments.csv（保留“评论”列即可）。
"""
import csv, argparse
from collections import Counter, defaultdict

POS = ["好","不错","满意","喜欢","舒服","舒适","轻便","轻","软","透气","划算","实惠",
       "好看","颜值","时尚","推荐","值得","快","及时","精细","结实","稳","回弹","明显",
       "耐磨","强","安全","划算","回购","好评","到位","优秀","开心","惊喜"]
NEG = ["差","失望","生气","磨脚","磨破","开胶","质量问题","硬","震脚","重","拖沓","打滑",
       "滑","异味","刺鼻","起皱","旧","薄","夹","疼","窄","小","偏","偏大","偏小","不准",
       "麻烦","慢","压皱","溢胶","线头","一般","中规中矩","慎","不值","降价","不平衡","少",
       "缺乏","没有","磨","破","划痕"]
NEGATION = ["不","没","无","未","别","莫","非","毫不","没有"]
DEGREE = {"很":1.6,"非常":1.8,"特别":1.8,"太":1.9,"极其":2.0,"挺":1.3,"比较":1.2,
          "有点":0.7,"稍微":0.7,"基本":0.6,"严重":1.8,"还算":1.0}
# 评价维度词典（维度 -> 触发词）
ASPECTS = {
    "尺码/鞋楦": ["尺码","偏小","偏大","偏窄","宽脚","鞋楦","标准码","正常码","拍大","拍小"],
    "脚感/缓震": ["缓震","回弹","震脚","软","硬","支撑","鞋垫","踩","脚感","膝盖"],
    "外观做工": ["颜值","好看","时尚","颜色","色差","款式","做工","线头","溢胶","起皱","显旧","精细"],
    "质量耐久": ["开胶","质量","耐磨","结实","破","异味","材质","划痕","厚重","重"],
    "物流/包装": ["物流","发货","包装","快递","到货","第二天","三天"],
    "价格体验": ["价格","实惠","划算","不值","降价","回购","价位"],
    "客服售后": ["客服","换货","退换","回复","售后","催"],
    "抓地/安全": ["打滑","抓地","防滑","摔跤","雨天"],
}
WINDOW = 4  # 否定/程度词向前作用窗口（字符）


def load_comments(path):
    with open(path, encoding="utf-8-sig") as f:
        return [r["评论"].strip() for r in csv.DictReader(f) if r.get("评论")]


def hit_words(text, words):
    """返回文本中命中的词列表（长词优先，避免“不错”被“不”先吃掉）"""
    found, used = [], [False]*len(text)
    for w in sorted(words, key=len, reverse=True):
        start = 0
        while True:
            i = text.find(w, start)
            if i < 0:
                break
            if not any(used[i:i+len(w)]):
                found.append((i, w))
                for k in range(i, i+len(w)):
                    used[k] = True
            start = i + len(w)
    return sorted(found)


PUNCT = "，。！？；、,.!?;~… \n"

def score(text):
    """返回 (情感分, 命中明细[(词, 极性, 权重, 位置)])"""
    detail, total = [], 0.0
    # 程度副词覆盖区间（避免“严重”里的“重”被当成负面词重复计分）
    degree_spans = []
    for d in DEGREE:
        start = 0
        while True:
            j = text.find(d, start)
            if j < 0:
                break
            degree_spans.append((j, j + len(d)))
            start = j + len(d)
    in_degree = lambda i: any(a <= i < b for a, b in degree_spans)

    for i, w in hit_words(text, POS + NEG):
        if len(w) == 1 and in_degree(i):
            continue
        polarity = 1 if w in POS else -1
        # 否定/程度只在“最近一个标点之后”的局部窗口内生效
        seg_start = i
        for k in range(i - 1, max(-1, i - WINDOW - 1), -1):
            if text[k] in PUNCT:
                break
            seg_start = k
        window = text[seg_start:i]
        weight = 1.0
        for d, mul in DEGREE.items():
            if d in window:
                weight *= mul
        flipped = any(n in window for n in NEGATION)
        if flipped:
            polarity *= -1
        total += polarity * weight
        detail.append((w, polarity, weight, i, flipped))
    return total, detail


def aspects_of(text, polarity):
    out = []
    for asp, kws in ASPECTS.items():
        if any(k in text for k in kws):
            out.append((asp, polarity))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="demo_comments.csv")
    ap.add_argument("--product", default="某品牌跑鞋（演示）")
    args = ap.parse_args()
    comments = load_comments(args.input)

    results, pos_kw, neg_asp, pos_asp = [], Counter(), Counter(), Counter()
    for c in comments:
        s, det = score(c)
        label = "好评" if s > 0.5 else "差评" if s < -0.5 else "中评"
        results.append((c, label, s))
        for w, pol, wt, _, flipped in det:
            if pol > 0 and s > 0.5:
                pos_kw[("不" + w) if flipped else w] += 1
        for asp, pol in aspects_of(c, s):
            if pol < 0:
                neg_asp[asp] += 1
            elif pol > 0:
                pos_asp[asp] += 1

    n = len(comments)
    cnt = Counter(r[1] for r in results)

    print("=" * 58)
    print(f"  商品评论情感分析报告 · {args.product}")
    print("=" * 58)
    print(f"  评论样本 {n} 条 | 好评 {cnt['好评']} 条({cnt['好评']/n*100:.0f}%)  "
          f"中评 {cnt['中评']} 条  差评 {cnt['差评']} 条({cnt['差评']/n*100:.0f}%)")
    print("-" * 58)
    print("  好评高频词：" + "、".join(f"{w}×{c}" for w, c in pos_kw.most_common(8)))
    print("  差评集中维度：" + "、".join(f"{w}×{c}" for w, c in neg_asp.most_common(5)))
    print("-" * 58)
    print("  差评原文（按情感分最负面）：")
    for c, label, s in sorted(results, key=lambda x: x[2])[:5]:
        if label == "差评":
            print(f"   - {c}")
    print("=" * 58)

    # 结论建议
    suggestions = []
    for asp, c in neg_asp.most_common(3):
        if asp == "尺码/鞋楦":
            suggestions.append("尺码问题突出：商品详情页增加尺码建议表/真人试穿反馈，客服话术前置提醒")
        elif asp == "质量耐久":
            suggestions.append("质量负面集中：抽检对应批次，重点排查开胶、异味问题，差评用户主动售后召回")
        elif asp == "脚感/缓震":
            suggestions.append("缓震预期管理：区分通勤/慢跑/竞速使用场景，避免宣传过度导致体验落差")
        elif asp == "抓地/安全":
            suggestions.append("抓地差评涉及安全：核实湿地防滑数据，页面补充湿滑路面提示")
        elif asp == "物流/包装":
            suggestions.append("物流包装波动：更换/加强对应仓配线路，升级鞋盒外箱防护")
        elif asp == "价格体验":
            suggestions.append("价保体验：为降价期订单主动补差或发券，降低“买贵”差评")
        elif asp == "外观做工":
            suggestions.append("做工细节：加强溢胶/线头品控，色差问题补充实拍图")
        elif asp == "客服售后":
            suggestions.append("售后响应：差评工单优先处理，设置退换时效承诺")

    L = [f"# {args.product} · 评论情感分析报告\n",
         f"- 样本量：{n} 条",
         f"- 好评率：**{cnt['好评']/n*100:.1f}%**（{cnt['好评']} 条），中评 {cnt['中评']} 条，差评 {cnt['差评']} 条\n",
         "## 用户夸什么", "、".join(f"{w}（{c}）" for w, c in pos_kw.most_common(8)) or "无",
         "\n## 用户骂什么（差评维度分布）"]
    for asp, c in neg_asp.most_common():
        L.append(f"- {asp}：{c} 条")
    L += ["\n## 典型差评"] + [f"> {c}" for c, l, s in
          sorted(results, key=lambda x: x[2])[:5] if l == "差评"]
    L += ["\n## 运营改进建议"] + [f"{i+1}. {s}" for i, s in enumerate(suggestions)]
    with open("sentiment_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("  已导出：sentiment_report.md")

    write_html(args.product, n, cnt, pos_kw, neg_asp, results, suggestions)
    print("  已导出：sentiment_report.html")


def write_html(product, n, cnt, pos_kw, neg_asp, results, suggestions):
    pct = lambda k: cnt[k] / n * 100 if n else 0
    worst = [c for c, l, s in sorted(results, key=lambda x: x[2]) if l == "差评"][:5]
    max_asp = max([c for _, c in neg_asp.most_common()] + [1])
    pos_tags = "".join(
        f'<span class="tag pos">{w}<b>×{c}</b></span>' for w, c in pos_kw.most_common(10)) or '<span class="dim">无</span>'
    asp_rows = "".join(
        f'<div class="arow"><span class="alab">{asp}</span>'
        f'<span class="abar"><i style="width:{c/max_asp*100:.0f}%"></i></span>'
        f'<span class="anum">{c} 条</span></div>'
        for asp, c in neg_asp.most_common()) or '<p class="dim">本期无明显差评集中维度</p>'
    bad_cards = "".join(f'<div class="bad">“{c}”</div>' for c in worst) or '<p class="dim">无典型差评</p>'
    sug_items = "".join(f"<li>{s}</li>" for s in suggestions) or "<li>差评维度平稳，保持现有品控与服务节奏</li>"

    html = TEMPLATE.replace("{{PRODUCT}}", product) \
        .replace("{{N}}", str(n)) \
        .replace("{{POS_PCT}}", f"{pct('好评'):.1f}") \
        .replace("{{MID_PCT}}", f"{pct('中评'):.1f}") \
        .replace("{{NEG_PCT}}", f"{pct('差评'):.1f}") \
        .replace("{{POS_N}}", str(cnt["好评"])) \
        .replace("{{MID_N}}", str(cnt["中评"])) \
        .replace("{{NEG_N}}", str(cnt["差评"])) \
        .replace("{{POS_W}}", f"{pct('好评'):.2f}") \
        .replace("{{MID_W}}", f"{pct('中评'):.2f}") \
        .replace("{{NEG_W}}", f"{pct('差评'):.2f}") \
        .replace("{{POS_TAGS}}", pos_tags) \
        .replace("{{ASP_ROWS}}", asp_rows) \
        .replace("{{BAD_CARDS}}", bad_cards) \
        .replace("{{SUG}}", sug_items)
    with open("sentiment_report.html", "w", encoding="utf-8") as f:
        f.write(html)


TEMPLATE = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>评论情感分析报告</title><style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:"Microsoft YaHei",sans-serif;background:#f4f5f7;color:#222;padding:24px 14px;line-height:1.6}
.box{max-width:880px;margin:0 auto;background:#fff;border-radius:14px;padding:28px 30px;box-shadow:0 1px 4px rgba(0,0,0,.07)}
h1{font-size:20px}.s{color:#999;font-size:12.5px;margin:4px 0 20px}
h2{font-size:15.5px;color:#b30000;margin:24px 0 12px;padding-left:10px;border-left:4px solid #b30000}
.cards{display:flex;gap:14px;flex-wrap:wrap}
.sc{flex:1;min-width:130px;border:1px solid #eee;border-radius:10px;padding:16px;text-align:center}
.sc .v{font-size:30px;font-weight:800}.sc .l{font-size:12.5px;color:#888;margin-top:2px}.sc .c{font-size:11.5px;color:#aaa}
.g{color:#1aad4a}.r{color:#e03636}.gr{color:#999}
.stack{display:flex;height:14px;border-radius:7px;overflow:hidden;margin:6px 0 4px}
.stack i{display:block;height:100%}
.legend{font-size:12px;color:#888}
.legend b{font-weight:700}
.tag{display:inline-block;font-size:12.5px;background:#f2faf4;color:#178a42;border:1px solid #d5efdd;border-radius:16px;padding:4px 12px;margin:0 6px 8px 0}
.tag b{font-weight:800;margin-left:2px}
.arow{display:flex;align-items:center;margin-bottom:9px;font-size:13px}
.alab{width:96px;flex:none;color:#444}
.abar{flex:1;background:#f6f6f6;border-radius:5px;height:15px;margin:0 10px;overflow:hidden}
.abar i{display:block;height:100%;background:linear-gradient(90deg,#e86a6a,#d82a2a);border-radius:5px}
.anum{width:44px;flex:none;text-align:right;color:#d82a2a;font-weight:700}
.bad{background:#fdf4f4;border-left:3px solid #e08080;border-radius:0 8px 8px 0;padding:10px 14px;font-size:13.5px;margin-bottom:9px;color:#5a3333}
ol{padding-left:20px;font-size:13.5px}ol li{margin-bottom:8px}
.dim{color:#aaa;font-size:13px}
.back{display:inline-block;margin-top:24px;color:#b30000;font-size:13px;text-decoration:none}
.foot{color:#bbb;font-size:11.5px;margin-top:18px;border-top:1px solid #f0f0f0;padding-top:12px}
@media(max-width:560px){.box{padding:20px 16px}.sc .v{font-size:25px}.alab{width:78px;font-size:12px}}
</style></head><body><div class="box">
<h1>{{PRODUCT}} · 评论情感分析报告</h1>
<div class="s">样本量 {{N}} 条 · 词典法分析（正/负词典 + 否定窗口反转 + 程度副词加权 + 维度归集）· 演示数据</div>

<div class="cards">
  <div class="sc"><div class="v g">{{POS_PCT}}%</div><div class="l">好评</div><div class="c">{{POS_N}} 条</div></div>
  <div class="sc"><div class="v gr">{{MID_PCT}}%</div><div class="l">中评</div><div class="c">{{MID_N}} 条</div></div>
  <div class="sc"><div class="v r">{{NEG_PCT}}%</div><div class="l">差评</div><div class="c">{{NEG_N}} 条</div></div>
</div>
<div class="stack"><i style="width:{{POS_W}}%;background:#1aad4a"></i><i style="width:{{MID_W}}%;background:#c4c4c4"></i><i style="width:{{NEG_W}}%;background:#e03636"></i></div>
<div class="legend"><span style="color:#1aad4a">■</span> 好评 <b>{{POS_W}}%</b> ｜ <span style="color:#c4c4c4">■</span> 中评 <b>{{MID_W}}%</b> ｜ <span style="color:#e03636">■</span> 差评 <b>{{NEG_W}}%</b></div>

<h2>用户夸什么</h2>
<div>{{POS_TAGS}}</div>

<h2>差评集中在哪些维度</h2>
{{ASP_ROWS}}

<h2>典型差评（情感分最负面）</h2>
{{BAD_CARDS}}

<h2>运营改进建议</h2>
<ol>{{SUG}}</ol>

<a class="back" href="index.html">← 返回项目首页</a>
<div class="foot">数据为演示评论，仅展示分析框架与报告形态；真实使用导入电商后台导出的评论 CSV 即可，词典可按品类持续扩充。</div>
</div></body></html>"""


if __name__ == "__main__":
    main()
