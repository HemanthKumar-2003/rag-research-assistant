from rag_assistant.database.schema import (
    initialize_database,
)


def main() -> None:
    initialize_database()

    print(
        "Database initialised successfully."
    )


if __name__ == "__main__":
    main()