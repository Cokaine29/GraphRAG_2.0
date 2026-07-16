"""
Unit tests for the router module.
"""

import pytest
from src.router import QueryRouter, RouteType

def test_route_type_enum():
    assert RouteType.LOCAL.value == "LOCAL"
    assert RouteType.GLOBAL.value == "GLOBAL"
    assert RouteType.HYBRID.value == "HYBRID"

def test_router_parse_json():
    router = QueryRouter()
    
    # Test clean JSON
    clean_json = '{"route_type": "LOCAL", "confidence": 0.9, "reasoning": "test"}'
    res = router._parse_json(clean_json)
    assert res.get("route_type") == "LOCAL"
    assert res.get("confidence") == 0.9
    
    # Test markdown fenced JSON
    fenced_json = '```json\n{"route_type": "GLOBAL", "confidence": 0.8}\n```'
    res = router._parse_json(fenced_json)
    assert res.get("route_type") == "GLOBAL"
    assert res.get("confidence") == 0.8
    
    # Test empty/malformed
    empty_res = router._parse_json('')
    assert isinstance(empty_res, dict)
