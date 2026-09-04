from datetime import datetime, timezone
from email import policy
from email.message import Message
from email.parser import BytesParser

from app.schemas.email import (
    EmailAttachment,
    EmailMessage,
)


class EmailParser:
    """
    Converts raw RFC822 email messages into the application's
    canonical EmailMessage schema.
    """

    def parse(
        self,
        raw_message: bytes,
        message_id: str,
    ) -> EmailMessage:
        """
        Parse a raw email into EmailMessage.
        """

        if not raw_message:
            raise ValueError(
                "Cannot parse an empty email message."
            )

        message = BytesParser(
            policy=policy.default
        ).parsebytes(raw_message)

        return EmailMessage(
            id=message_id,
            sender=self._extract_sender(message),
            subject=self._extract_subject(message),
            received_at=self._extract_received_at(
                message
            ),
            body=self._extract_body(message),
            attachments=self._extract_attachments(
                message
            ),
        )

    @staticmethod
    def _extract_sender(
        message: Message,
    ) -> str:
        sender = message.get("From")

        if not sender:
            return "Not stated"

        return str(sender).strip()

    @staticmethod
    def _extract_subject(
        message: Message,
    ) -> str:
        subject = message.get("Subject")

        if not subject:
            return "Not stated"

        return str(subject).strip()

    @staticmethod
    def _extract_received_at(
        message: Message,
    ) -> datetime:
        """
        Extract the email Date header.

        If the header is missing or malformed, use UTC now as a
        processing-safe fallback rather than failing the entire
        message.
        """

        date_header = message.get("Date")

        if date_header:
            try:
                from email.utils import parsedate_to_datetime

                parsed_date = parsedate_to_datetime(
                    date_header
                )

                if parsed_date.tzinfo is None:
                    parsed_date = parsed_date.replace(
                        tzinfo=timezone.utc
                    )

                return parsed_date
            except (TypeError, ValueError):
                pass

        return datetime.now(timezone.utc)

    @classmethod
    def _extract_body(
        cls,
        message: Message,
    ) -> str:
        """
        Extract the preferred text body.

        For multipart emails:
        - prefer text/plain
        - fall back to text/html
        """

        if not message.is_multipart():
            return cls._decode_part(message)

        plain_text = None
        html_text = None

        for part in message.walk():
            if part.is_multipart():
                continue

            content_type = part.get_content_type()

            if content_type == "text/plain":
                decoded = cls._decode_part(part)

                if decoded.strip():
                    plain_text = decoded

            elif content_type == "text/html":
                decoded = cls._decode_part(part)

                if decoded.strip():
                    html_text = decoded

        if plain_text:
            return plain_text.strip()

        if html_text:
            return html_text.strip()

        return ""

    @staticmethod
    def _decode_part(
        part: Message,
    ) -> str:
        """
        Decode a MIME message part safely.
        """

        try:
            content = part.get_content()

            if isinstance(content, str):
                return content
        except (LookupError, UnicodeDecodeError):
            pass

        payload = part.get_payload(
            decode=True
        )

        if isinstance(payload, bytes):
            charset = (
                part.get_content_charset()
                or "utf-8"
            )

            return payload.decode(
                charset,
                errors="replace",
            )

        if isinstance(payload, str):
            return payload

        return ""

    @staticmethod
    def _extract_attachments(
        message: Message,
    ) -> list[EmailAttachment]:
        """
        Extract attachment metadata without storing attachment
        bytes.

        Actual attachment persistence will be implemented in the
        mailbox/document ingestion layer.
        """

        attachments: list[EmailAttachment] = []

        attachment_index = 0

        for part in message.walk():
            if part.is_multipart():
                continue

            filename = part.get_filename()

            if not filename:
                continue

            attachment_index += 1

            payload = part.get_payload(
                decode=True
            )

            size_bytes = (
                len(payload)
                if isinstance(payload, bytes)
                else 0
            )

            content_type = (
                part.get_content_type()
            )

            attachments.append(
                EmailAttachment(
                    id=f"attachment-{attachment_index}",
                    filename=filename,
                    content_type=content_type,
                    size_bytes=size_bytes,
                    is_pdf=(
                        content_type
                        == "application/pdf"
                        or filename.lower().endswith(
                            ".pdf"
                        )
                    ),
                )
            )

        return attachments