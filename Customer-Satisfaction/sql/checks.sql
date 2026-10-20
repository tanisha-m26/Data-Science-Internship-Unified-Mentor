-- Check for missing ratings
SELECT COUNT(*) AS missing_ratings
FROM tickets
WHERE customer_satisfaction_rating IS NULL;

-- Check for invalid age
SELECT COUNT(*) AS invalid_age
FROM tickets
WHERE customer_age < 0 OR customer_age > 120;

-- Check ticket status distribution
SELECT ticket_status, COUNT(*) 
FROM tickets
GROUP BY ticket_status;
-- Check for duplicate tickets
SELECT ticket_id, COUNT(*) AS count
FROM tickets
GROUP BY ticket_id
HAVING COUNT(*) > 1;

