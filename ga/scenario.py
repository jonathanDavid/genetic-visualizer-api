"""Deterministic retail scenario generator for the "pickup" problem.

A scenario is a small city snapshot: M stores placed on a 2-D km map around a
customer at the origin, each stocking a subset of P grocery products (with
per-store prices in COP cents varying ±15% around a catalog base price and a
stock count), plus a shopping list of N needed products that is **guaranteed
coverable** (every needed product is stocked by at least one store).

Everything is derived from a single seeded numpy RNG in a fixed draw order,
so the same ``(seed, stores, products, needs)`` always yields an identical
scenario — across processes, across the REST endpoint and GA runs.

Distances are euclidean; travel time assumes 30 km/h urban speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

URBAN_SPEED_KMH = 30.0

# Real-ish Colombian grocery catalog: (name, base price in COP cents).
CATALOG: list[tuple[str, int]] = [
    ("Arroz blanco 500g", 320_000),
    ("Leche entera 1L", 480_000),
    ("Huevos AA x12", 950_000),
    ("Pan tajado familiar", 620_000),
    ("Queso campesino 250g", 890_000),
    ("Café molido 250g", 1_250_000),
    ("Azúcar blanca 1kg", 460_000),
    ("Aceite de girasol 1L", 1_390_000),
    ("Frijol cargamanto 500g", 780_000),
    ("Panela redonda 500g", 340_000),
    ("Pasta espagueti 500g", 390_000),
    ("Atún en agua 170g", 720_000),
    ("Lentejas 500g", 440_000),
    ("Harina de maíz 1kg", 510_000),
    ("Chocolate de mesa 250g", 680_000),
    ("Mantequilla 250g", 980_000),
    ("Yogur natural 1L", 840_000),
    ("Avena en hojuelas 500g", 560_000),
    ("Sal refinada 500g", 180_000),
    ("Galletas saladas taco", 430_000),
    ("Jugo de naranja 1L", 650_000),
    ("Plátano verde x kg", 290_000),
    ("Tomate chonto x kg", 410_000),
    ("Cebolla cabezona x kg", 360_000),
    ("Papa pastusa x kg", 270_000),
    ("Pollo entero x kg", 1_190_000),
    ("Carne de res molida 500g", 1_480_000),
    ("Jabón de baño x3", 870_000),
    ("Papel higiénico x4", 1_090_000),
    ("Detergente en polvo 1kg", 1_230_000),
]

_NAME_PREFIXES = [
    "Mercado", "Super", "Tienda", "Bodega",
    "Autoservicio", "Almacén", "Minimarket", "Plaza",
]
_NAME_SUFFIXES = [
    "Norte", "Sur", "Costa", "Centro", "Andino", "del Parque",
    "La Esquina", "El Trébol", "Popular", "Granada", "San Fernando", "La Colina",
]


@dataclass(slots=True)
class InventoryItem:
    sku: str
    name: str
    price_cents: int
    stock: int


@dataclass(slots=True)
class Store:
    id: str
    name: str
    x: float
    y: float
    distance_km: float
    inventory: list[InventoryItem] = field(default_factory=list)

    def stocks(self, sku: str) -> bool:
        return any(item.sku == sku for item in self.inventory)

    def price_of(self, sku: str) -> int | None:
        for item in self.inventory:
            if item.sku == sku:
                return item.price_cents
        return None


@dataclass(slots=True)
class NeededProduct:
    sku: str
    name: str
    qty: int = 1


class ScenarioValidationError(ValueError):
    """Raised when an explicit scenario is malformed or not coverable."""


@dataclass(slots=True)
class Scenario:
    seed: int
    customer: tuple[float, float]
    stores: list[Store]
    shopping_list: list[NeededProduct]

    # ------------------------------------------------------------------ #
    # Convenience lookups used by PickupProblem and the baselines
    # ------------------------------------------------------------------ #
    def stores_stocking(self, sku: str) -> list[int]:
        return [i for i, s in enumerate(self.stores) if s.stocks(sku)]

    def price_cents(self, store_idx: int, sku: str) -> int:
        price = self.stores[store_idx].price_of(sku)
        if price is None:
            raise KeyError(f"store {store_idx} does not stock {sku}")
        return price

    def to_dict(self) -> dict[str, Any]:
        """Contract shape for ``GET /api/scenario``."""
        return {
            "seed": self.seed,
            "customer": {"x": self.customer[0], "y": self.customer[1]},
            "stores": [
                {
                    "id": s.id,
                    "name": s.name,
                    "x": s.x,
                    "y": s.y,
                    "distanceKm": s.distance_km,
                    "inventory": [
                        {
                            "sku": it.sku,
                            "name": it.name,
                            "priceCents": it.price_cents,
                            "stock": it.stock,
                        }
                        for it in s.inventory
                    ],
                }
                for s in self.stores
            ],
            "shoppingList": [
                {"sku": n.sku, "name": n.name, "qty": n.qty} for n in self.shopping_list
            ],
        }


def travel_minutes(km: float) -> float:
    return km / URBAN_SPEED_KMH * 60.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScenarioValidationError(message)


def scenario_from_dict(data: Any) -> Scenario:
    """Build a :class:`Scenario` from an explicit client-supplied object.

    Used verbatim (no seeding) so the retail app can optimize a real order
    against real inventory. Validates structure and, crucially, that every
    shopping-list SKU is stocked by at least one store — otherwise the order
    is not coverable and we surface a clear 422 instead of a silent bad plan.
    Customer defaults to the origin; each store's ``distanceKm`` is derived
    from the customer position.
    """
    _require(isinstance(data, dict), "scenario must be an object")

    cust = data.get("customer") or {"x": 0.0, "y": 0.0}
    _require(isinstance(cust, dict), "scenario.customer must be an object")
    cx, cy = float(cust.get("x", 0.0)), float(cust.get("y", 0.0))

    raw_stores = data.get("stores")
    _require(isinstance(raw_stores, list) and len(raw_stores) >= 1,
             "scenario.stores must be a non-empty list")

    stores: list[Store] = []
    seen_ids: set[str] = set()
    for k, raw in enumerate(raw_stores):
        _require(isinstance(raw, dict), f"stores[{k}] must be an object")
        sid = str(raw.get("id") or f"s{k + 1}")
        _require(sid not in seen_ids, f"duplicate store id {sid!r}")
        seen_ids.add(sid)
        x, y = float(raw.get("x", 0.0)), float(raw.get("y", 0.0))
        raw_inv = raw.get("inventory") or []
        _require(isinstance(raw_inv, list), f"stores[{k}].inventory must be a list")
        inventory: list[InventoryItem] = []
        for j, it in enumerate(raw_inv):
            _require(isinstance(it, dict), f"stores[{k}].inventory[{j}] must be an object")
            sku = it.get("sku")
            _require(bool(sku), f"stores[{k}].inventory[{j}].sku is required")
            price = it.get("priceCents")
            _require(price is not None, f"stores[{k}].inventory[{j}].priceCents is required")
            price = int(price)
            _require(price >= 0, f"stores[{k}].inventory[{j}].priceCents must be >= 0")
            inventory.append(
                InventoryItem(
                    sku=str(sku),
                    name=str(it.get("name") or sku),
                    price_cents=price,
                    stock=int(it.get("stock", 0)),
                )
            )
        stores.append(
            Store(
                id=sid,
                name=str(raw.get("name") or sid),
                x=x,
                y=y,
                distance_km=round(math.hypot(x - cx, y - cy), 3),
                inventory=inventory,
            )
        )

    raw_list = data.get("shoppingList")
    _require(isinstance(raw_list, list) and len(raw_list) >= 1,
             "scenario.shoppingList must be a non-empty list")

    shopping_list: list[NeededProduct] = []
    for k, raw in enumerate(raw_list):
        _require(isinstance(raw, dict), f"shoppingList[{k}] must be an object")
        sku = raw.get("sku")
        _require(bool(sku), f"shoppingList[{k}].sku is required")
        qty = int(raw.get("qty", 1))
        _require(qty >= 1, f"shoppingList[{k}].qty must be >= 1")
        name = raw.get("name")
        if not name:  # borrow the catalog name from any store that stocks it
            name = next(
                (it.name for s in stores for it in s.inventory if it.sku == sku),
                str(sku),
            )
        shopping_list.append(NeededProduct(sku=str(sku), name=str(name), qty=qty))

    # Coverage: the heart of the validation — an uncoverable order is a 422.
    scenario = Scenario(seed=-1, customer=(cx, cy), stores=stores, shopping_list=shopping_list)
    uncoverable = [n.sku for n in shopping_list if not scenario.stores_stocking(n.sku)]
    _require(
        not uncoverable,
        "shopping list is not coverable: no store stocks "
        + ", ".join(sorted(set(uncoverable))),
    )
    return scenario


def generate_scenario(
    seed: int = 42,
    stores: int = 6,
    products: int = 12,
    needs: int = 5,
) -> Scenario:
    """Build a deterministic scenario from ``(seed, stores, products, needs)``.

    ``needs`` is clamped to ``products``. The shopping list is always
    coverable: any needed product missing from every store is injected into a
    seeded-random store's inventory.
    """
    stores = int(stores)
    products = int(min(products, len(CATALOG)))
    needs = int(min(needs, products))
    rng = np.random.default_rng(seed)

    # --- store identities -------------------------------------------------
    combo_count = len(_NAME_PREFIXES) * len(_NAME_SUFFIXES)
    combos = rng.choice(combo_count, size=stores, replace=False)
    names = [
        f"{_NAME_PREFIXES[c // len(_NAME_SUFFIXES)]} {_NAME_SUFFIXES[c % len(_NAME_SUFFIXES)]}"
        for c in combos
    ]

    # --- store placement (annulus around the customer at the origin) ------
    radii = rng.uniform(0.8, 8.0, size=stores)
    angles = rng.uniform(0.0, 2.0 * math.pi, size=stores)
    xs = np.round(radii * np.cos(angles), 3)
    ys = np.round(radii * np.sin(angles), 3)

    # --- product catalog slice --------------------------------------------
    catalog = CATALOG[:products]
    skus = [f"p{i + 1:02d}" for i in range(products)]

    # --- per-store inventories --------------------------------------------
    store_objs: list[Store] = []
    for i in range(stores):
        stocked = np.nonzero(rng.random(products) < 0.55)[0]
        if stocked.size < 3:  # every store carries at least a few items
            extra = rng.choice(
                np.setdiff1d(np.arange(products), stocked),
                size=3 - stocked.size,
                replace=False,
            )
            stocked = np.sort(np.concatenate([stocked, extra]))
        inventory = [
            InventoryItem(
                sku=skus[p],
                name=catalog[p][0],
                price_cents=int(round(catalog[p][1] * rng.uniform(0.85, 1.15))),
                stock=int(rng.integers(1, 25)),
            )
            for p in stocked.tolist()
        ]
        store_objs.append(
            Store(
                id=f"s{i + 1}",
                name=names[i],
                x=float(xs[i]),
                y=float(ys[i]),
                distance_km=float(round(math.hypot(xs[i], ys[i]), 3)),
                inventory=inventory,
            )
        )

    # --- shopping list, guaranteed coverable -------------------------------
    needed = rng.choice(products, size=needs, replace=False)
    shopping_list = [
        NeededProduct(sku=skus[p], name=catalog[p][0], qty=1) for p in needed.tolist()
    ]
    for p in needed.tolist():
        sku = skus[p]
        if not any(s.stocks(sku) for s in store_objs):
            host = store_objs[int(rng.integers(0, stores))]
            host.inventory.append(
                InventoryItem(
                    sku=sku,
                    name=catalog[p][0],
                    price_cents=int(round(catalog[p][1] * rng.uniform(0.85, 1.15))),
                    stock=int(rng.integers(1, 25)),
                )
            )

    return Scenario(
        seed=seed,
        customer=(0.0, 0.0),
        stores=store_objs,
        shopping_list=shopping_list,
    )
