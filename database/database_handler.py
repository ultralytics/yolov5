import os
import subprocess
import json
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, String, Boolean, Date, Integer, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base

from utils.general import LOGGER


class DBConfigSQLAlchemy:
    Base = declarative_base()

    def __init__(self):
        self.engine = None
        self.session_maker = None

    def _get_db_access_token(self, client_id):
        # Authenticate using Managed Identity (MSI)
        try:
            command = ["az", "login", "--identity", "--username", client_id]
            subprocess.check_call(command)
        except subprocess.CalledProcessError as e:
            print("Error during 'az login --identity':", e)
            raise e

        # Execute Azure CLI command to get the access token
        command = ["az", "account", "get-access-token", "--resource-type", "oss-rdbms"]
        output = subprocess.check_output(command)

        # Parse the output to retrieve the access token
        access_token = json.loads(output)["accessToken"]

        return access_token

    def _get_database_connection_string(self):
        """
        Gets the connection string for the PostgreSQL database.

        The method first checks if the code is running locally. If it is,
        it retrieves the PostgreSQL credentials from environment variables.
        Otherwise, it loads the credentials from a JSON file in the project root.
        """
        if os.environ.get('POSTGRES_USER'):
            # Local run, use environment variables for credentials
            db_url = f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@{os.environ['POSTGRES_HOST']}/{os.environ['POSTGRES_DB']}"
        else:
            # Production run, load credentials from the JSON file
            with open('database.json') as f:
                config = json.load(f)

            # Retrieve values from the JSON
            hostname = config["hostname"]
            username = config["username"]
            database_name = config["database_name"]
            client_id = config["client_id"]

            # Get the password using the client_id from a secure source
            password = self._get_db_access_token(client_id)

            db_url = f"postgresql://{username}:{password}@{hostname}/{database_name}"

        return db_url

    def create_connection(self):
        try:
            # Create the engine
            db_url = self._get_database_connection_string()
            self.engine = create_engine(db_url)
            self.session_maker = sessionmaker(bind=self.engine)

            LOGGER.info(f"Successfully created database sessionmaker.")

        except SQLAlchemyError as e:
            # Handle any exceptions that occur during connection creation
            LOGGER.error(f"Error creating database sessionmaker: {str(e)}")
            raise e

    def get_session(self):
        if self.session_maker is None:
            raise RuntimeError("SessionMaker has not been created. Call create_connection() first.")

        return self.session_maker()

    def close_connection(self):
        try:
            # Dispose the engine
            self.engine.dispose()
        except SQLAlchemyError as e:
            LOGGER.error(f"Error disposing the database engine: {str(e)}")
            raise e

    @contextmanager
    def managed_session(self):
        session = self.get_session()
        try:
            yield session  # This line yields the 'session' to the with block.
            session.commit()  # Executed when the with block completes
        except SQLAlchemyError as e:
            session.rollback()
            raise e
        finally:
            session.close()
