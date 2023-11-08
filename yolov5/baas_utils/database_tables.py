from sqlalchemy import Column, String, Boolean, DateTime, Integer, Float
from .database_handler import DBConfigSQLAlchemy


class ImageProcessingStatus(DBConfigSQLAlchemy.Base):
    __tablename__ = 'image_processing_status'
    __table_args__ = {'schema': 'private_schema_blur'}  # Add the schema here

    image_customer_name = Column(String, primary_key=True)
    image_upload_date = Column(DateTime, primary_key=True)
    image_filename = Column(String, primary_key=True)
    processing_status = Column(String)


class DetectionInformation(DBConfigSQLAlchemy.Base):
    __tablename__ = 'detection_information'
    __table_args__ = {'schema': 'private_schema_blur'}  # Add the schema here

    id = Column(Integer, primary_key=True)
    image_customer_name = Column(String)
    image_upload_date = Column(DateTime)
    image_filename = Column(String)
    has_detection = Column(Boolean)
    class_id = Column(Integer)
    x_norm = Column(Float)
    y_norm = Column(Float)
    w_norm = Column(Float)
    h_norm = Column(Float)
    image_width = Column(Integer)
    image_height = Column(Integer)
    run_id = Column(String)


class BatchRunInformation(DBConfigSQLAlchemy.Base):
    __tablename__ = 'batch_run_information'
    __table_args__ = {'schema': 'private_schema_blur'}  # Add the schema here

    run_id = Column(String, primary_key=True)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    trained_yolo_model = Column(String)
    success = Column(Boolean)
    error_code = Column(String)
