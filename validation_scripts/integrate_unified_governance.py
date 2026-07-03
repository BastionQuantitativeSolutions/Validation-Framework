"""
Integration Script: Add Unified Governance to Live Trading
===========================================================

This script adds the unified_governance module as an additional check
in the live trading system (main_loop.py).

The unified_governance provides 10 key governance gates:
1. Holiday Gate
2. UNKNOWN Regime Gate (CRITICAL)
3. Session+Regime Gate
4. Confidence Gate
5. Momentum Gate
6. Factors Gate
7. Daily Cap Gate
8. Loss Streak Gate
9. Session Cap Gate
10. Full Pass-through Gate
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

INTEGRATION_MARKER = """
# UNIFIED_GOVERNANCE_INTEGRATION_START
# This section integrates the unified_governance module into the live trading system
# Unified governance provides 10 key gates that mirror live trading exactly
try:
    from CORE_MODULES.core.unified_governance import governance_check as unified_gov_check
    _UNIFIED_GOV_AVAILABLE = True
except ImportError:
    _UNIFIED_GOV_AVAILABLE = False
    unified_gov_check = None

def _apply_unified_governance(pair, tf, direction, regime, confidence, session, factors_count):
    \"\"\"Apply unified governance check if available.\"\"\"
    if not _UNIFIED_GOV_AVAILABLE or unified_gov_check is None:
        return True, "UNIFIED_GOV_UNAVAILABLE"
    
    result = unified_gov_check(
        pair=pair,
        tf=tf,
        direction=direction,
        regime=regime if regime else "UNKNOWN",
        confidence=confidence,
        session=session or "OTHER",
        factors_count=factors_count,
    )
    return result["allowed"], result.get("reason", "UNKNOWN")
# UNIFIED_GOVERNANCE_INTEGRATION_END
"""


def main():
    main_loop_path = Path(__file__).parents[2] / "core/runtime/main_loop.py"

    if not main_loop_path.exists():
        print(f"ERROR: {main_loop_path} not found")
        return

    content = main_loop_path.read_text()

    if "UNIFIED_GOVERNANCE_INTEGRATION_START" in content:
        print("Unified governance integration already present")
        return

    lines = content.split("\n")

    insert_idx = None
    for i, line in enumerate(lines):
        if line.startswith("from core.governance import entry_governor as deg"):
            insert_idx = i + 1
            break

    if insert_idx is None:
        print("Could not find import location in main_loop.py")
        return

    lines.insert(insert_idx, INTEGRATION_MARKER)

    main_loop_path.write_text("\n".join(lines))
    print(f"Added unified_governance integration to {main_loop_path}")
    print("\nNext steps:")
    print("1. Run validation: python CORE_MODULES/validation/validate_unified_system.py")
    print("2. Test in paper trading mode")
    print("3. Deploy to live when ready")


if __name__ == "__main__":
    main()
