import unittest
from unittest.mock import patch


class _FakePhyEngineLibOk:
    def __init__(self, _lib_path: str):
        pass

    def can_simulate_status_save(self) -> bool:
        return True

    def simulate_status_save(self, *, status_save: dict, analyze_type: int, **_kwargs) -> dict:
        # Return a minimal StatusSave-like object with Statistics filled.
        assert isinstance(status_save, dict)
        assert analyze_type in (1, 4)
        return {
            "Elements": [
                {
                    "Identifier": "R1",
                    "Label": "R1",
                    "ModelID": "Resistor",
                    "Statistics": {"电压": 1.0, "电流": 0.001, "功率": 0.001},
                }
            ],
            "Wires": [],
        }


class _FakePhyEngineLibFail(_FakePhyEngineLibOk):
    def simulate_status_save(self, *args, **kwargs):
        from pe_sim import PESimError

        raise PESimError("adapter boom")


class TestStatusSaveAdapter(unittest.TestCase):
    def test_status_save_adapter_branch_formats_output(self):
        import tools

        status_save = {"Elements": [{"Identifier": "R1", "ModelID": "Resistor", "Properties": {"电阻": 1000}}], "Wires": []}

        class _Cfg:
            sim_timeout_sec = 0

        with patch.object(tools, "ensure_phyengine_lib", return_value="/tmp/libphyengine.so"), patch.object(
            tools, "PhyEngineLib", _FakePhyEngineLibOk
        ):
            out = tools.simulate_status_save_with_phyengine(
                text="simulate dc",
                status_save=status_save,
                phy_engine_cfg=_Cfg(),
                config_base_dir=".",
            )
        self.assertIn("Simulation result", out)
        self.assertIn("R1", out)
        self.assertIn("V≈", out)

    def test_adapter_error_is_returned_when_fallback_mapper_also_fails(self):
        import tools

        status_save = {"Elements": [{"Identifier": "X1", "ModelID": "UnknownThing", "Properties": {}}], "Wires": []}

        class _Cfg:
            sim_timeout_sec = 0

        with patch.object(tools, "ensure_phyengine_lib", return_value="/tmp/libphyengine.so"), patch.object(
            tools, "PhyEngineLib", _FakePhyEngineLibFail
        ):
            out = tools.simulate_status_save_with_phyengine(
                text="simulate dc",
                status_save=status_save,
                phy_engine_cfg=_Cfg(),
                config_base_dir=".",
            )
        self.assertIn("adapter", out.lower())
        self.assertIn("fallback", out.lower())


if __name__ == "__main__":
    unittest.main()
