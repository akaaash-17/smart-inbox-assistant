from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Smart Inbox Assistant"
    environment: str = "development"
    version: str = "0.1.0"

    ai_provider: str = "ollama"
    ai_model: str = "llama3.2"

    database_url: str = ""

    mailbox_host: str = ""
    mailbox_port: int = 993
    mailbox_username: str = ""
    mailbox_password: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()