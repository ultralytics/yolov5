import subprocess
import json
from contextlib import contextmanager
import time

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError, DatabaseError
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timedelta

from yolov5.utils.general import LOGGER

class DBConfigSQLAlchemy:
    Base = declarative_base()

    def __init__(self, db_username, db_hostname, db_name):
        self.engine = None
        self.session_maker = None
        self.db_username = db_username
        self.db_hostname = db_hostname
        self.db_name = db_name
        self.access_token = None
        self.token_expiration_time = None
        self.token_renewal_margin = timedelta(minutes=5)
        self.retry_count = 3
        self.retry_delay = 5  # seconds between retries

    def _get_db_access_token(self):
        with open('database.json') as f:
            config = json.load(f)

        # Retrieve values from the JSON
        client_id = config["client_id"]  # TODO get this from the Managed Identity name in code

        # Authenticate using Managed Identity (MSI)
        try:
            command = ["az", "login", "--identity", "--username", client_id]
            subprocess.check_call(command)
        except subprocess.CalledProcessError as e:
            LOGGER.error("Error during 'az login --identity': {e}")
            raise e

        # Execute Azure CLI command to get the access token
        command = ["az", "account", "get-access-token", "--resource-type", "oss-rdbms"]
        output = subprocess.check_output(command)

        # Parse the output to retrieve the access token and expiration time
        token_info = json.loads(output)
        self.access_token = token_info["accessToken"]
        expires_on_str = token_info["expiresOn"]
        # Convert the expiration time string to a datetime object
        token_expiration_time = datetime.strptime(expires_on_str, "%Y-%m-%d %H:%M:%S.%f")
        self.token_expiration_time = token_expiration_time - self.token_renewal_margin

    def _get_db_connection_string(self):
        self._get_db_access_token()
        db_url = f"postgresql://{self.db_username}:{self.access_token}@{self.db_hostname}/{self.db_name}"
        return db_url

    def _get_session(self):
        if self.session_maker is None:
            raise RuntimeError("SessionMaker has not been created. Call create_connection() first.")

        return self.session_maker()

    def create_connection(self):
        try:
            # Create the engine
            db_url = self._get_db_connection_string()
            self.engine = create_engine(db_url)
            self.session_maker = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

            LOGGER.info(f"Successfully created database sessionmaker.")

        except SQLAlchemyError as e:
            # Handle any exceptions that occur during connection creation
            LOGGER.error(f"Error creating database sessionmaker: {str(e)}")
            raise e

    @contextmanager
    def managed_session(self):
        for retry in range(self.retry_count):
            self._validate_token_status()

            session = self._get_session()
            try:
                yield session  # This line yields the 'session' to the with block.
                session.commit()  # Executed when the with block completes
                break  # Exit the loop if successful
            except SQLAlchemyError as e:
                # Roll back any uncommitted changes within current session to maintain data integrity.
                session.rollback()
                raise e
            except DatabaseError as e:
                # You can add a sleep here before the next retry
                if retry < self.retry_count - 1:
                    LOGGER.error(f"Error with the connection to the database, retry after {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    raise e
            session.close()

    def close_connection(self):
        try:
            self.engine.dispose()
        except SQLAlchemyError as e:
            LOGGER.error(f"Error disposing the database engine: {str(e)}")
            raise e

    def _validate_token_status(self):
        if self.token_expiration_time < datetime.now():
            self._get_db_access_token()
            LOGGER.info("Token for database renewed.")
