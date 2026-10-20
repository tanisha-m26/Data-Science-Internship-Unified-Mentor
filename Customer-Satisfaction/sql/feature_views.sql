-- Aggregate features for ML
CREATE VIEW ticket_features AS
SELECT
    ticket_id,
    customer_age,
    customer_gender,
    product_purchased,
    ticket_type,
    ticket_priority,
    ticket_channel,
    EXTRACT(EPOCH FROM (time_to_resolution - first_response_time))/3600 AS resolution_hours,
    customer_satisfaction_rating
FROM tickets;


