"""C-MAPSS veri setinin kolon şeması. Dosyalarda başlık satırı yoktur."""

UNIT = "unit"
CYCLE = "cycle"
RUL = "RUL"

INDEX_COLS = [UNIT, CYCLE]
SETTING_COLS = [f"op{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS