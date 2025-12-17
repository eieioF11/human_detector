import gymnasium as gym

gym.register(
    id="NavigationEnv-v0",  # 環境名
    entry_point="env.navigation_env:NavigationEnv",  # PATH.ファイル名（拡張子なし）: クラス名
    disable_env_checker=True,
)

gym.register(
    id="NavigationEnv-single_action",  # 環境名
    entry_point="env.navigation_env_single_action:NavigationEnv",  # PATH.ファイル名（拡張子なし）: クラス名
    disable_env_checker=True,
)

gym.register(
    id="NavigationEnv-custom",  # 環境名
    entry_point="env.navigation_env_custom:NavigationEnv",  # PATH.ファイル名（拡張子なし）: クラス名
    disable_env_checker=True,
)

gym.register(
    id="NavigationEnv-test",  # 環境名
    entry_point="env.navigation_env_test:NavigationEnv",  # PATH.ファイル名（拡張子なし）: クラス名
    disable_env_checker=True,
)

gym.register(
    id="NavigationEnv-v2",  # 環境名
    entry_point="env.navigation_env_v2:NavigationEnv",  # PATH.ファイル名（拡張子なし）: クラス名
    disable_env_checker=True,
)

gym.register(
    id="NavigationEnv-v3",  # 環境名
    entry_point="env.navigation_env_v3:NavigationEnv",  # PATH.ファイル名（拡張子なし）: クラス名
    disable_env_checker=True,
)

gym.register(
    id="NavigationEnv-v3_2",  # 環境名
    entry_point="env.navigation_env_v3_2:NavigationEnv",  # PATH.ファイル名（拡張子なし）: クラス名
    disable_env_checker=True,
)

gym.register(
    id="NavigationEnv-v4",  # 環境名
    entry_point="env.navigation_env_v4:NavigationEnv",  # PATH.ファイル名（拡張子なし）: クラス名
    disable_env_checker=True,
)
