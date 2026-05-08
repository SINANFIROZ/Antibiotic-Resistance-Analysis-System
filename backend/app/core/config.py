from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    project_name: str = 'AMR Intelligence Platform'
    environment: str = 'development'
    api_v1_prefix: str = '/api/v1'
    backend_host: str = '0.0.0.0'
    backend_port: int = 8000
    database_url: str = 'sqlite+aiosqlite:///./amr_platform.db'
    sync_database_url: str = 'sqlite:///./amr_platform.db'
    jwt_secret_key: str = 'change-me-in-production'
    jwt_algorithm: str = 'HS256'
    access_token_expire_minutes: int = 60
    refresh_token_expire_minutes: int = 60 * 24 * 7
    cors_origins: list[str] = Field(default_factory=lambda: ['http://localhost:3000'])
    rate_limit: str = '100/minute'
    auto_create_schema: bool = True
    seed_reference_catalog: bool = True
    initial_admin_email: str = 'admin@amr.local'
    initial_admin_password: str = 'ChangeMe123!'

    def model_post_init(self, __context) -> None:
        if self.environment != 'development' and self.initial_admin_password == 'ChangeMe123!':
            raise ValueError('INITIAL_ADMIN_PASSWORD must be overridden outside development environments')

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    @property
    def report_storage_dir(self) -> Path:
        return self.project_root / 'storage' / 'reports'

    @property
    def dataset_storage_dir(self) -> Path:
        return self.project_root / 'storage' / 'datasets'

    @property
    def legacy_model_dirs(self) -> list[Path]:
        return [self.project_root / 'Models', self.project_root / 'models']


@lru_cache
def get_settings() -> Settings:
    return Settings()
