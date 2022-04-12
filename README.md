# pv-usecase

## Config options

| key                | type                                                 | description                                               | required |
|--------------------|------------------------------------------------------|-----------------------------------------------------------|----------|
| `energy_src_id`    | string                                               | ID of source providing energy data.                       | yes      |
| `weather_src_id`   | string                                               | ID of source providing weather forcast data.              | yes      |
| `logger_level`     | string                                               | `info`, `warning` (default), `error`, `critical`, `debug` | no       |
| `selectors`        | array[object{"name": string, "args": array[string]}] | Define selectors to distinguish between data sources.     | no       |
| `history_power_td` | integer                                              |                                                           | no       |
| `weather_dim`      | integer                                              |                                                           | no       |
| `data_path`        | string                                               | Path to reward and model files. Default: "/opt/data"      | no       |
