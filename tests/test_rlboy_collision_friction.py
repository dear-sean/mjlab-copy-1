"""Tests for RL_BOY whole-body collision friction."""

from mjlab.asset_zoo.robots.RL_BOY.rlboy_constants import (
  FULL_COLLISION,
  get_spec,
)


def test_all_rlboy_collision_geoms_have_tangential_friction() -> None:
  spec = get_spec()
  FULL_COLLISION.edit_spec(spec)

  collision_geoms = [geom for geom in spec.geoms if geom.name.endswith("_collision")]
  assert collision_geoms
  assert all(geom.condim >= 3 for geom in collision_geoms)
  assert all(geom.friction[0] > 0.0 for geom in collision_geoms)

  foot_geoms = [geom for geom in collision_geoms if "_foot" in geom.name]
  nonfoot_geoms = [geom for geom in collision_geoms if "_foot" not in geom.name]
  assert foot_geoms
  assert nonfoot_geoms
  assert all(geom.condim == 6 and geom.friction[0] == 1.0 for geom in foot_geoms)
  assert all(geom.condim == 3 and geom.friction[0] == 0.6 for geom in nonfoot_geoms)
