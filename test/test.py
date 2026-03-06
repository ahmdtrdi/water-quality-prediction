import os

def test_config_exists():
    assert os.path.exists("config/base.yaml")

def test_source_package():
  
    import src
    assert src is not None