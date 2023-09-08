import subprocess
import json
from contextlib import contextmanager
import time
from threading import Thread

from sqlalchemy import create_engine, Column, String, Boolean, Date, Integer, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timedelta

from utils.general import LOGGER

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
        self.token_renewal_margin = timedelta(minutes=5)  # Set the token renewal margin

    def _get_db_access_token(self, client_id):
        # Authenticate using Managed Identity (MSI)
        try:
            command = ["az", "login", "--identity", "--username", client_id]
            subprocess.check_call(command)
        except subprocess.CalledProcessError as e:
            LOGGER.info("Error during 'az login --identity': {e}")
            raise e

        # Execute Azure CLI command to get the access token
        command = ["az", "account", "get-access-token", "--resource-type", "oss-rdbms"]
        output = subprocess.check_output(command)

        # Parse the output to retrieve the access token and expiration time
        token_info = json.loads(output)
        self.access_token = token_info["accessToken"]
        expires_on_str = token_info["expiresOn"]
        # Convert the expiration time string to a datetime object
        self.token_expiration_time = datetime.strptime(expires_on_str, "%Y-%m-%d %H:%M:%S.%f")

    def _get_db_connection_string(self):
        # Get the connection string for the PostgreSQL database.

        # Production run, load credentials from the JSON file
        with open('database.json') as f:
            config = json.load(f)

        # Retrieve values from the JSON
        client_id = config["client_id"]  # TODO get this from the Managed Identity name in code

        # Get the password using the client_id from a secure source
        self._get_db_access_token(client_id)

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
            self.session_maker = sessionmaker(bind=self.engine)

            LOGGER.info(f"Successfully created database sessionmaker.")

        except SQLAlchemyError as e:
            # Handle any exceptions that occur during connection creation
            LOGGER.info(f"Error creating database sessionmaker: {str(e)}")
            raise e

    @contextmanager
    def managed_session(self):
        session = self._get_session()
        try:
            yield session  # This line yields the 'session' to the with block.
            session.commit()  # Executed when the with block completes
        except SQLAlchemyError as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def close_connection(self):
        try:
            # Dispose the engine
            self.engine.dispose()
        except SQLAlchemyError as e:
            LOGGER.info(f"Error disposing the database engine: {str(e)}")
            raise e

    @staticmethod
    def extract_upload_date(path):
        parts = path.split('/')
        image_filename = parts[-1]
        date_time_str = parts[-2]

        try:
            image_upload_date = datetime.strptime(date_time_str, "%Y-%m-%d_%H:%M:%S")
            image_upload_date = image_upload_date.strftime("%Y-%m-%d")
        except ValueError as e:
            LOGGER.info(f"Invalid folder structure, can not retrieve date: {date_time_str}")
            raise e

        return image_filename, image_upload_date

    def _renew_token_periodically(self):
        while True:
            # Calculate the time to renew the token
            renew_time = self.token_expiration_time - self.token_renewal_margin

            # Sleep until the renewal time
            sleep_time = (renew_time - datetime.now()).total_seconds()
            if sleep_time > 0:
                LOGGER.info(f"Sleeping for {sleep_time} seconds until token renewal.")
                time.sleep(sleep_time)

            # Renew the token after the sleep
            self._get_db_access_token()
            LOGGER.info("Token for database renewed.")

    def start_token_renewal_thread(self):
        # TODO will this have conflicts with the threads used by YOLOv5?
        # Start a thread to renew the token periodically
        token_renewal_thread = Thread(target=self._renew_token_periodically)
        token_renewal_thread.daemon = True
        token_renewal_thread.start()
