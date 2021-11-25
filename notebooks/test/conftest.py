def pytest_collectstart(collector):
    collector.skip_compare += 'application/vnd.jupyter.widget-view+json',
