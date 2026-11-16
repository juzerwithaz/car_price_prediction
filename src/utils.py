"""
AutoPrice Compass - Utility functions
Helper routines for INR currency formatting, deal meter calculation, and UI styling.
"""

from typing import Tuple, Dict, Any


def format_inr(amount: float) -> str:
    """Format a number into Indian Rupee representation (e.g. ₹5,45,000 or ₹14.20 Lakh)."""
    if amount is None or amount < 0:
        return "₹0"
    
    amount_int = int(round(amount))
    
    # Format with Indian numbering system
    s = str(amount_int)
    if len(s) <= 3:
        formatted = s
    else:
        last3 = s[-3:]
        remaining = s[:-3]
        groups = []
        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.insert(0, remaining)
        formatted = ",".join(groups) + "," + last3
        
    # Also add Lakhs / Crores string for quick readability
    if amount >= 10_000_000:
        cr = amount / 10_000_000
        return f"₹{formatted} ({cr:.2f} Cr)"
    elif amount >= 100_000:
        lakh = amount / 100_000
        return f"₹{formatted} ({lakh:.2f} Lakh)"
    else:
        return f"₹{formatted}"


def format_lakhs(amount: float) -> str:
    """Return compact amount in Lakhs, e.g. ₹5.45 L."""
    if amount >= 10_000_000:
        return f"₹{amount / 10_000_000:.2f} Cr"
    elif amount >= 100_000:
        return f"₹{amount / 100_000:.2f} L"
    return f"₹{amount:,.0f}"


def evaluate_deal(predicted_price: float, asking_price: float) -> Dict[str, Any]:
    """
    Compare seller's asking price against the ML predicted fair price.
    Returns status label, badge tone, difference in INR, and percent variance.
    """
    if asking_price <= 0 or predicted_price <= 0:
        return {
            "status": "Invalid Asking Price",
            "verdict": "neutral",
            "difference": 0.0,
            "percent": 0.0,
            "message": "Please enter a valid asking price greater than zero."
        }
    
    ratio = asking_price / predicted_price
    diff = asking_price - predicted_price
    pct = (diff / predicted_price) * 100.0

    if ratio <= 0.88:
        return {
            "status": "🔥 Exceptional Deal (Underpriced)",
            "verdict": "success",
            "difference": diff,
            "percent": pct,
            "badge_color": "#10B981",
            "message": f"Listed {abs(pct):.1f}% below fair market price. High buyer value!"
        }
    elif ratio <= 1.05:
        return {
            "status": "✅ Fair Market Price",
            "verdict": "info",
            "difference": diff,
            "percent": pct,
            "badge_color": "#3B82F6",
            "message": f"Within reasonable valuation tolerance ({-abs(pct):.1f}% to +{pct:.1f}%)."
        }
    elif ratio <= 1.20:
        return {
            "status": "⚠️ Slightly Overpriced",
            "verdict": "warning",
            "difference": diff,
            "percent": pct,
            "badge_color": "#F59E0B",
            "message": f"Priced {pct:.1f}% higher than typical valuation. Room to negotiate."
        }
    else:
        return {
            "status": "🚨 Significantly Overpriced",
            "verdict": "error",
            "difference": diff,
            "percent": pct,
            "badge_color": "#EF4444",
            "message": f"Listed {pct:.1f}% above fair value. Reconsider or negotiate strongly."
        }
