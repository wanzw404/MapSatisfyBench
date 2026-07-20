#!/usr/bin/env python3
"""CLI tool tester — 批量测试 app/core/tools 下所有工具。

Usage:
    python -m app.cli.test_tools                    # 测试所有工具
    python -m app.cli.test_tools --list             # 列出所有工具
    python -m app.cli.test_tools --tool search_poi  # 测试单个工具
    python -m app.cli.test_tools --sandbox          # 全部使用沙箱模式
"""
import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.tools.manager import tools as ALL_TOOLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── 每个工具的测试参数（key = tool.name） ──
TOOL_TEST_CASES = {
    "ainative_kuake_search": {
        "query": "北京今天天气怎么样",
        "query_write": "北京天气,北京今日天气",
    },
    "current_location_query": {
        "location": "116.481488,39.990464",
        "radius": "1000",
        "extensions": "all",
    },
    "fuel_payment": {
        "poi_id": "B0FFHL3VW5",
        "oil_num": "95",
        "oil_gun": "3",
        "oil_price": "200",
    },
    "get_navigation": {
        "start_lon": "116.4074",
        "start_lat": "39.9042",
        "end_lon": "116.4034",
        "end_lat": "39.9151",
        "start_name": "天安门广场",
        "end_name": "故宫博物院",
        "mode": ["驾车", "公交"],
        "route_type": "34",
    },
    "get_rgeo": {
        "x": "116.4074",
        "y": "39.9042",
    },
    "get_route_traffic_info": {
        "poi_id": "B0IADL1JQK",
    },
    "get_sequential_navigation": {
        "points": [
            {"lon": 116.4074, "lat": 39.9042, "name": "天安门广场"},
            {"lon": 116.3974, "lat": 39.9180, "name": "王府井"},
            {"lon": 116.4034, "lat": 39.9151, "name": "故宫博物院"},
        ],
    },
    "get_taxi_route_plan": {
        "start_lat": "39.9042",
        "start_lon": "116.4074",
        "end_lat": "39.9151",
        "end_lon": "116.4034",
        "start_name": "天安门广场",
        "end_name": "故宫博物院",
        "end_poi_id": "B000A8URXB",
    },
    "get_weather": {
        "location": "北京",
        "date": "2025-09-11",
        "whole": True,
    },
    "optimize_visit_order": {
        "start_name": "天安门广场",
        "start_lat": "39.9042",
        "start_lon": "116.4074",
        "via_names": "王府井,北海公园",
        "via_points": "116.3974,39.9180;116.3880,39.9250",
        "end_name": "故宫博物院",
        "end_lat": "39.9151",
        "end_lon": "116.4034",
        "mode": "fixed_end",
    },
    "restaurant_group_buy": {
        "poi_id": "B0FFJGDE5L",
    },
    "restaurant_reservation": {
        "poi_id": "B0FFJGDE5L",
        "table_type": "包厢",
        "people_num": "4",
        "order_time": "2025-09-11 18:00",
    },
    "route_station_info": {
        "keywords": "来广营路口西",
        "city": "110000",
        "offset": "20",
        "page": "1",
    },
    "scenic_ticket_transaction": {
        "datetime": "2025-10-01",
        "personnum": "2",
        "sightname": "故宫博物院",
        "sightpoiid": "B000A8URXB",
        "ticketsession": "上午场",
        "tickettype": "成人票",
    },
    "search_around_poi": {
        "center_name": "天安门广场",
        "query": "餐厅",
        "range": 5000,
        "rating_range": "[3,10]",
        "sort_rule": "2",
        "x": "116.4074",
        "y": "39.9042",
    },
    "search_flights": {
        "date": "2025-07-15",
        "from_city": "北京市",
        "to_city": "上海市",
        "filter_strategy": "全天智能过滤",
    },
    "search_poi": {
        "query": "北京大学",
        "cur_adcode": "110000",
    },
    "search_poi_along_route": {
        "start_x": "116.4074",
        "start_y": "39.9042",
        "end_x": "116.4034",
        "end_y": "39.9151",
        "keywords": "餐厅",
        "route_type": "0",
        "range": "10000",
    },
    "search_poi_around_multipoints": {
        "query": "酒店",
        "pois": [
            {"x": 116.4074, "y": 39.9042},
            {"x": 116.3974, "y": 39.9180},
            {"x": 116.4034, "y": 39.9151},
        ],
        "range_meters": 5000,
        "need_centrality_filter": "1",
        "rating_range": "[4,10]",
        "hotel_price_range": "[200,800]",
    },
    "search_products_by_poiid": {
        "poiid": "B0K39CGDOX",
    },
    "search_train_tickets": {
        "date": "2025-07-11",
        "from_city": "北京市",
        "from_city_adcode": "110000",
        "from_lat": "39.9042",
        "from_lon": "116.4074",
        "to_city": "上海市",
        "to_city_adcode": "310000",
        "to_lat": "31.230525",
        "to_lon": "121.473667",
    },
    "transaction_service": {
        "itemname": "故宫门票",
        "itemtype": "门票",
        "orderobject": "故宫博物院",
    },
    "search_user_action_summary": {
        "adiu": "qcdaglj98f668dgiehhgg9e4ccaca4",
    },
    "search_user_profile": {
        "adiu": "qcdaglj98f668dgiehhgg9e4ccaca4",
    },
}


def _run_single(tool, sandbox: bool) -> dict:
    """测试单个工具，返回统计信息。"""
    name = tool.name
    args = dict(TOOL_TEST_CASES.get(name, {}))
    if "sandbox" in [p for p in tool.args_schema.model_fields]:
        args["sandbox"] = sandbox

    logger.info("=" * 70)
    logger.info(f"[测试工具] {name}")
    logger.info(f"[入参] {json.dumps(args, ensure_ascii=False, default=str)}")

    start = time.time()
    try:
        result = tool.invoke(args)
        elapsed_ms = int((time.time() - start) * 1000)
        result_str = str(result)
        logger.info(f"[耗时] {elapsed_ms} ms")
        logger.info(f"[结果] {result_str[:500]}{'...' if len(result_str) > 500 else ''}")
        return {"name": name, "status": "success", "elapsed_ms": elapsed_ms}
    except Exception as e:
        elapsed_ms = int((time.time() - start) * 1000)
        logger.error(f"[耗时] {elapsed_ms} ms")
        logger.error(f"[异常] {type(e).__name__}: {e}")
        logger.debug(traceback.format_exc())
        return {"name": name, "status": "error", "elapsed_ms": elapsed_ms, "error": str(e)}


def _list_tools():
    print("可用工具列表：")
    for tool in ALL_TOOLS:
        has_case = "✓" if tool.name in TOOL_TEST_CASES else "✗"
        print(f"  {has_case} {tool.name}")


def main():
    parser = argparse.ArgumentParser(description="测试 app/core/tools 下所有工具")
    parser.add_argument("--tool", type=str, default=None, help="指定单个工具名测试")
    parser.add_argument("--sandbox", action="store_true", default=False, help="使用沙箱模式")
    parser.add_argument("--list", action="store_true", default=False, help="列出所有工具")
    args = parser.parse_args()

    if args.list:
        _list_tools()
        return

    # 确定要测试的工具列表
    if args.tool:
        targets = [t for t in ALL_TOOLS if t.name == args.tool]
        if not targets:
            logger.error(f"未找到工具: {args.tool}")
            logger.info("可用工具:")
            for t in ALL_TOOLS:
                logger.info(f"  - {t.name}")
            sys.exit(1)
    else:
        targets = ALL_TOOLS

    logger.info(f"{'沙箱' if args.sandbox else '真实 API'}模式，共 {len(targets)} 个工具待测试")

    stats = []
    for tool in targets:
        stat = _run_single(tool, args.sandbox)
        stats.append(stat)

    # 汇总
    success = sum(1 for s in stats if s["status"] == "success")
    failed = sum(1 for s in stats if s["status"] == "error")
    logger.info("=" * 70)
    logger.info(f"测试完成 | 成功: {success} | 失败: {failed} | 总计: {len(stats)}")
    if failed:
        logger.info("失败列表：")
        for s in stats:
            if s["status"] == "error":
                logger.info(f"  - {s['name']}: {s.get('error', '')}")


if __name__ == "__main__":
    main()
