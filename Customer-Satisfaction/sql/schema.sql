-- Schema for customer support ticket dataset

CREATE TABLE tickets (
    ticket_id INT PRIMARY KEY,
    customer_name VARCHAR(100),
    customer_email VARCHAR(150),
    customer_age INT,
    customer_gender VARCHAR(20),
    product_purchased VARCHAR(100),
    date_of_purchase DATE,
    ticket_type VARCHAR(50),
    ticket_subject VARCHAR(200),
    ticket_description TEXT,
    ticket_status VARCHAR(50),
    resolution TEXT,
    ticket_priority VARCHAR(50),
    ticket_channel VARCHAR(50),
    first_response_time TIMESTAMP,
    time_to_resolution TIMESTAMP,
    customer_satisfaction_rating INT
);
