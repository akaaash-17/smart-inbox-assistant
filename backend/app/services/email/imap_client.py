from dataclasses import dataclass
from email.message import Message
from imaplib import IMAP4_SSL
from typing import Iterator


@dataclass
class RawEmail:
    """
    Represents a raw email retrieved from an IMAP mailbox.
    """

    message_id: str
    raw_message: bytes


class IMAPClient:
    """
    Lightweight IMAP client responsible only for mailbox access.

    Parsing and business logic are intentionally kept outside this
    class so that the mailbox layer can be replaced or mocked easily.
    """

    def __init__(
        self,
        host: str,
        port: int = 993,
        username: str = "",
        password: str = "",
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password

        self.connection: IMAP4_SSL | None = None

    def connect(self) -> None:
        """
        Establish an SSL connection and authenticate.
        """

        if not self.host:
            raise ValueError(
                "IMAP host must be configured."
            )

        if not self.username:
            raise ValueError(
                "IMAP username must be configured."
            )

        if not self.password:
            raise ValueError(
                "IMAP password must be configured."
            )

        self.connection = IMAP4_SSL(
            self.host,
            self.port,
        )

        self.connection.login(
            self.username,
            self.password,
        )

    def select_mailbox(
        self,
        mailbox: str = "INBOX",
    ) -> None:
        """
        Select the mailbox to read.
        """

        connection = self._require_connection()

        status, _ = connection.select(mailbox)

        if status != "OK":
            raise RuntimeError(
                f"Unable to select mailbox: {mailbox}"
            )

    def fetch_unseen(
        self,
    ) -> Iterator[RawEmail]:
        """
        Fetch unseen email messages from the selected mailbox.
        """

        connection = self._require_connection()

        status, data = connection.search(
            None,
            "UNSEEN",
        )

        if status != "OK":
            raise RuntimeError(
                "Unable to search for unseen emails."
            )

        message_ids = data[0].split()

        for message_id in message_ids:
            status, message_data = connection.fetch(
                message_id,
                "(RFC822)",
            )

            if status != "OK":
                continue

            raw_message = self._extract_raw_message(
                message_data
            )

            if raw_message is None:
                continue

            yield RawEmail(
                message_id=message_id.decode(
                    errors="replace"
                ),
                raw_message=raw_message,
            )

    def close(self) -> None:
        """
        Close the selected mailbox and IMAP connection.
        """

        if self.connection is None:
            return

        try:
            self.connection.close()
        except Exception:
            pass

        try:
            self.connection.logout()
        except Exception:
            pass

        self.connection = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()

    def _require_connection(self) -> IMAP4_SSL:
        if self.connection is None:
            raise RuntimeError(
                "IMAP connection has not been established."
            )

        return self.connection

    @staticmethod
    def _extract_raw_message(
        message_data,
    ) -> bytes | None:
        """
        Extract RFC822 bytes from an IMAP FETCH response.
        """

        for item in message_data:
            if not isinstance(item, tuple):
                continue

            if len(item) < 2:
                continue

            payload = item[1]

            if isinstance(payload, bytes):
                return payload

        return None