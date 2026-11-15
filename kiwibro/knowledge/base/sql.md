# SQL Reference

## SELECT: reading rows from a table
`SELECT name, price FROM products;` returns the named columns from every row of the table. `SELECT *` returns all columns, and expressions work too: `SELECT price * 2 AS double_price FROM products;`. Reading data with SELECT does not change the table.

## WHERE: filtering rows
`SELECT * FROM products WHERE price > 100;` keeps only rows where the condition is true. Compare with `=`, `<>` or `!=`, `<`, `>`, `<=`, `>=`; combine with `AND`, `OR` and `NOT`, and use parentheses when mixing them. Text goes in single quotes: `WHERE name = 'tea'`.

## ORDER BY: sorting the result
`ORDER BY price` sorts ascending and `ORDER BY price DESC` descending; `ORDER BY category ASC, price DESC` sorts by the first column, then the second. Text sorting depends on the database collation. Without ORDER BY, the order of returned rows is undefined.

## LIMIT and OFFSET: taking part of the result
`SELECT * FROM logs ORDER BY id DESC LIMIT 10;` returns the first 10 rows. `LIMIT 10 OFFSET 20` skips 20 rows and returns the next 10. Paging needs a fixed ORDER BY, or rows can repeat or disappear between pages. MySQL also writes `LIMIT 20, 10` with the offset first; SQL Server uses `TOP` and `OFFSET ... FETCH`.

## DISTINCT: removing duplicate rows
`SELECT DISTINCT city FROM customers;` returns each value once. DISTINCT looks at the whole selected row, so `SELECT DISTINCT city, country` removes rows that are equal in both columns. `COUNT(DISTINCT city)` counts the unique values.

## GROUP BY: one result row per group
`SELECT category, COUNT(*) FROM products GROUP BY category;` collects rows with the same category into groups and runs the aggregates once per group. Every selected column must be grouped or aggregated. GROUP BY runs after WHERE, so WHERE filters rows before grouping.

## HAVING: filtering groups
`SELECT category, COUNT(*) AS n FROM products GROUP BY category HAVING COUNT(*) > 5;` keeps only groups that pass the test after aggregation. WHERE cannot use aggregate functions; HAVING can. Both may appear in one query: WHERE filters rows, then HAVING filters the groups.

## COUNT: counting rows
`COUNT(*)` counts rows, `COUNT(column)` counts rows where that column is not NULL, and `COUNT(DISTINCT column)` counts unique non-NULL values. COUNT never returns NULL; with no matching rows it returns 0.

## SUM, AVG, MIN and MAX: aggregate functions
`SUM(col)`, `AVG(col)`, `MIN(col)` and `MAX(col)` reduce many rows to one value and ignore NULLs, so AVG divides by the number of non-NULL values. MIN and MAX also work on text (alphabetical order) and dates. With no rows, SUM and AVG return NULL.

## INNER JOIN: rows that match in both tables
`SELECT o.id, c.name FROM orders o INNER JOIN customers c ON o.customer_id = c.id;` keeps only rows with a match on both sides. `JOIN` on its own means INNER JOIN. A join without ON pairs every row with every row, which is a cross join.

## LEFT JOIN: keeping all rows of the left table
`SELECT c.name, o.id FROM customers c LEFT JOIN orders o ON o.customer_id = c.id;` returns every customer, with NULL in the order columns when there is no matching order. This is the usual way to find missing records: add `WHERE o.id IS NULL`.

## RIGHT JOIN: keeping all rows of the right table
`SELECT c.name, o.id FROM orders o RIGHT JOIN customers c ON o.customer_id = c.id;` returns every row of the table on the right, with NULLs where the left table has no match. It is the mirror of a LEFT JOIN and can always be rewritten as one by swapping the tables.

## FULL OUTER JOIN: keeping rows from both sides
A FULL OUTER JOIN returns matched rows plus unmatched rows from both tables, filling the missing side with NULL. MySQL does not support it; there it is built from a LEFT JOIN and a RIGHT JOIN joined with UNION.

## CROSS JOIN: every combination of rows
`SELECT * FROM colours CROSS JOIN sizes;` pairs each row of one table with each row of the other, so 3 colours and 4 sizes give 12 rows. It is useful for generating combinations and is what a join with no ON condition produces anyway.

## Self join: a table joined to itself
`SELECT e.name, m.name AS manager FROM employees e JOIN employees m ON e.manager_id = m.id;` joins a table to itself using two aliases. It answers questions about rows related to other rows of the same table, such as managers, parent categories or earlier versions.

## Subquery: a query inside a query
A subquery is a SELECT in parentheses used as a value, a list or a table: `WHERE price > (SELECT AVG(price) FROM products)`. A subquery in FROM must have an alias in most databases: `FROM (SELECT ...) AS t`. One in the SELECT list must return a single value.

## Correlated subquery: a subquery that uses the outer row
A correlated subquery refers to a column of the current outer row, so it is evaluated again for each row: `SELECT name FROM products p WHERE price > (SELECT AVG(price) FROM products x WHERE x.category = p.category);`. It is clear but can be slow on large tables; a join often does the same job faster.

## EXISTS: testing whether a subquery returns rows
`SELECT * FROM customers c WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id);` keeps rows for which the subquery finds at least one row. `NOT EXISTS` keeps the opposite. It stops at the first match, so it is often faster than counting.

## INSERT INTO: adding rows
`INSERT INTO products (name, price) VALUES ('Tea', 5);` adds one row; `VALUES ('Tea', 5), ('Coffee', 7)` adds several. The column list sets the order of the values; columns left out get their DEFAULT or NULL. Leaving out the column list means giving every column in table order, which breaks when the table changes.

## UPDATE: changing existing rows
`UPDATE products SET price = 9 WHERE id = 3;` changes the named columns in matching rows only. Several columns are set in one statement: `SET price = price * 1.1, stock = stock - 1`. Run the same WHERE in a SELECT first to see which rows will change.

## UPDATE without WHERE: changes every row
`UPDATE products SET price = 0;` has no filter, so every row of the table is rewritten. The old values are gone unless the statement runs inside a transaction that is rolled back, or you have a backup. To change one row, put WHERE on the primary key and check the rows with a SELECT first.

## DELETE: removing rows
`DELETE FROM products WHERE id = 3;` removes the matching rows; the table, its columns and its indexes remain. The database logs each removed row so the change can be rolled back and fires delete triggers, which makes DELETE slower than TRUNCATE when clearing a whole table.

## DELETE without WHERE: empties the table
`DELETE FROM products;` removes every row of the table. The table and its structure stay, but the data is gone, and without a transaction or backup it cannot be restored. A mistaken condition in WHERE does the same, so test the condition with a SELECT before running the delete.

## TRUNCATE TABLE: emptying a table quickly
`TRUNCATE TABLE logs;` discards all rows at once without logging each one, so it is fast and does not fire row triggers. MySQL and SQL Server reset the auto-increment counter; PostgreSQL keeps it unless RESTART IDENTITY is given. In MySQL it commits immediately and cannot be rolled back; PostgreSQL and SQL Server allow it inside a transaction. SQLite has no TRUNCATE; it uses `DELETE FROM logs;`.

## CREATE TABLE: defining a table
`CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT NOT NULL, price REAL DEFAULT 0);` gives each column a name, a type and optional constraints. `CREATE TABLE IF NOT EXISTS` skips the statement when the table already exists, which is useful in setup scripts.

## Common SQL data types
Typical types are `INTEGER` for whole numbers, `REAL`, `FLOAT` and `DOUBLE` for approximate decimals, `DECIMAL(p, s)` for exact decimals such as money, `VARCHAR(n)` and `TEXT` for text, `DATE` and `TIMESTAMP` for dates, `BOOLEAN` (often stored as 0 and 1) and `BLOB` for raw bytes. Names differ between databases.

## INTEGER PRIMARY KEY and AUTOINCREMENT in SQLite
In SQLite, `id INTEGER PRIMARY KEY` makes the column an alias of the internal rowid, and inserting NULL or leaving it out assigns the next number. Adding `AUTOINCREMENT` guarantees that ids are never reused after a delete, at a small cost in speed and space.

## ALTER TABLE: changing a table's structure
`ALTER TABLE products ADD COLUMN stock INTEGER DEFAULT 0;` adds a column; `RENAME COLUMN old TO new`, `DROP COLUMN` and `RENAME TO` change or remove parts of a table. Adding a NOT NULL column without a default usually fails on a table that already has rows.

## DROP TABLE: deleting the table itself
`DROP TABLE products;` removes the table, every row in it, its indexes, triggers and permissions. Unlike DELETE and TRUNCATE, which keep the empty table, nothing is left to query afterwards. The data can be recovered only from a backup, unless the database allows DDL inside a transaction.

## Primary key: the unique identifier of a row
A primary key is one or more columns that identify each row uniquely, and each table has only one. Standard SQL forbids NULL in it; SQLite is looser and allows NULL in a primary key that is not an INTEGER, so add NOT NULL there. Databases index the key automatically. `PRIMARY KEY (a, b)` makes a composite key across two columns.

## Foreign key: linking two tables
A foreign key is a column that must match a primary key in another table: `FOREIGN KEY (customer_id) REFERENCES customers(id)`. The database then refuses orders for customers that do not exist, and `ON DELETE CASCADE` deletes the child rows when the parent goes. SQLite checks foreign keys only after `PRAGMA foreign_keys = ON;`.

## Index: speeding up lookups
An index is a sorted structure, usually a B-tree, on one or more columns so the database finds rows without scanning the whole table: `CREATE INDEX idx_name ON products(name);`. It speeds up filters, joins and sorts on those columns but takes space and slows INSERT, UPDATE and DELETE. Primary keys and UNIQUE columns are indexed automatically.

## UNIQUE constraint: no duplicate values
`email TEXT UNIQUE` or `UNIQUE (email)` stops two rows from holding the same value in that column, while still allowing NULLs (and in most databases several NULLs). The database also creates an index for it, so lookups on the column are fast. Use a primary key for the identity of a row and UNIQUE for other values.

## NOT NULL constraint: a column must have a value
`name TEXT NOT NULL` rejects any insert or update that leaves the column empty. It is separate from DEFAULT, which supplies a value when none is given; a column can have both. NOT NULL is the main way to stop missing data at the source.

## DEFAULT: the value used when none is given
`status TEXT DEFAULT 'new'` fills the column when an INSERT leaves it out. A default can be a constant or an expression such as `CURRENT_TIMESTAMP`. It applies only when the column is not mentioned; writing NULL explicitly stores NULL unless NOT NULL forbids it.

## CHECK constraint: a rule for values
`CHECK (price >= 0)` makes the database reject rows that break the rule, on insert and update. Checks can compare several columns of the same row: `CHECK (end_date >= start_date)`. They cannot look at other tables; that needs a trigger or a foreign key.

## NULL: the missing value
NULL means unknown or absent, and it is not 0 or an empty string. Any comparison with NULL gives unknown rather than true or false, so `WHERE price = NULL` matches nothing; use `IS NULL`. Aggregates ignore NULLs, except `COUNT(*)`, which counts rows.

## IS NULL and IS NOT NULL: testing for missing values
`WHERE email IS NULL` finds rows where the column has no value; `IS NOT NULL` finds rows where it does. These are the only correct tests, because `email = NULL` is unknown for every row and therefore never true.

## COALESCE: the first value that is not NULL
`COALESCE(nickname, first_name, 'guest')` returns the first argument that is not NULL. It is the portable way to give a fallback for missing data, and it also converts NULL to a known value before arithmetic, where NULL would otherwise spread through the result.

## LIKE: matching text with wildcards
`WHERE name LIKE 'tea%'` matches text starting with tea; `%` stands for any number of characters and `_` for exactly one. `LIKE '%tea%'` matches anywhere, and a literal `%` is escaped with an ESCAPE clause. Case sensitivity depends on the database and collation; PostgreSQL offers `ILIKE`, and `LOWER()` works everywhere.

## IN and NOT IN: matching a list of values
`WHERE status IN ('new', 'paid', 'sent')` is shorthand for a chain of OR tests, and `NOT IN` is its opposite. A NULL inside a NOT IN list makes the condition unknown for every row, so nothing matches; filter out NULLs first. IN also accepts a subquery that returns one column.

## BETWEEN: a range with two ends
`WHERE price BETWEEN 10 AND 20` is the same as `price >= 10 AND price <= 20`; both ends are included. For timestamps, a range such as `BETWEEN '2026-01-01' AND '2026-01-31'` leaves out most of the last day because of the time part, so use `>= start AND < next_day`.

## CASE: choosing a value per row
`SELECT name, CASE WHEN price < 10 THEN 'cheap' WHEN price < 100 THEN 'mid' ELSE 'dear' END AS band FROM products;` returns the first matching branch, or ELSE when none matches, and NULL when there is no ELSE. It works in SELECT, WHERE and ORDER BY. The simple form compares one value: `CASE status WHEN 'a' THEN 1 ELSE 0 END`.

## UNION and UNION ALL: stacking results
`SELECT email FROM customers UNION SELECT email FROM leads;` puts the rows of one result under the other; both queries need the same number of columns with compatible types, and the column names come from the first. UNION removes duplicate rows and is slower; UNION ALL keeps every row.

## AS: naming columns and tables in a query
`SELECT price * 2 AS double_price FROM orders AS o;` gives a result column or a table a shorter name. Table aliases are needed for self joins and make long queries readable; column aliases name computed values. Some databases accept a missing AS, but writing it is clearer.

## SQL statement order and execution order
A query is written `SELECT ... FROM ... WHERE ... GROUP BY ... HAVING ... ORDER BY ... LIMIT`. The database runs FROM and JOIN first, then WHERE, GROUP BY, HAVING, SELECT, DISTINCT, ORDER BY and LIMIT. That is why a column alias works in ORDER BY but not in WHERE.

## Aggregate query vs window function
An aggregate with GROUP BY collapses each group into one row. A window function keeps every row and adds a computed value: `ROW_NUMBER() OVER (PARTITION BY category ORDER BY price DESC)` numbers rows inside each category. `SUM(price) OVER (PARTITION BY category)` adds a group total next to each row without losing it.

## BEGIN, COMMIT and ROLLBACK: transactions
`BEGIN;` starts a transaction, and its changes are kept by `COMMIT;` or undone by `ROLLBACK;`. Until the commit, other connections usually do not see the changes. PostgreSQL also accepts `START TRANSACTION`; SQLite uses `BEGIN` and offers `BEGIN IMMEDIATE` to take the write lock straight away.

## ACID: the guarantees of a transaction
Atomicity: a transaction happens completely or not at all. Consistency: it moves the database from one valid state to another. Isolation: concurrent transactions do not see each other's unfinished work. Durability: once committed, the change survives a crash or power loss.

## First normal form (1NF)
A table is in 1NF when every column holds a single value: no lists, no repeated groups and no several values packed into one cell. A column of comma-separated phone numbers breaks it; a separate rows table fixes it.

## Second normal form (2NF)
A table is in 2NF when it is in 1NF and every non-key column depends on the whole primary key, not just part of a composite key. A product name stored in an order-lines table keyed by order and product belongs in the products table instead.

## Third normal form (3NF)
A table is in 3NF when it is in 2NF and no non-key column depends on another non-key column. Storing a city and also its postcode in the same table creates that dependency; the postcode belongs in a separate table keyed by city.

## View: a saved query used as a table
`CREATE VIEW cheap AS SELECT id, name, price FROM products WHERE price < 10;` stores the query, not the data. Selecting from the view runs the query each time, so it always shows current rows. Views simplify repeated queries and can hide columns; a materialised view stores results instead and needs refreshing.

## SQL injection: user input changing a query
SQL injection happens when a program builds SQL by joining strings with user input. Input `' OR '1'='1` turns `WHERE name = ''` into a condition that is always true, and input ending in `; DROP TABLE users` can destroy data where the driver allows several statements. The fix is parameterised queries, never concatenation.

## Parameterised queries and prepared statements
A parameterised query keeps SQL and values apart: `SELECT * FROM users WHERE email = ?` with the value sent separately, so the database treats it as data, never as SQL. Placeholders are `?`, `$1` or `:name` depending on the database and driver. Every value from a user, a file or a URL should go through a parameter.

## SQLite type affinity: flexible column types
SQLite has five affinities: INTEGER, REAL, TEXT, BLOB and NUMERIC. A declared type is mapped by its name, so `VARCHAR(20)` gets TEXT and `BIGINT` gets INTEGER. Any value can be stored in any column and is converted when possible; a STRICT table enforces the declared types instead.

## PRAGMA foreign_keys: turning on SQLite foreign keys
SQLite does not enforce foreign keys unless asked: `PRAGMA foreign_keys = ON;` switches the checks on for the current connection, and the command must be run every time a connection opens. Without it, a foreign key is only documentation and rows pointing to missing parents can be stored.

## Comments and string literals in SQL
`-- comment` runs to the end of the line and `/* comment */` marks a block. Text values use single quotes: `'tea'`. Double quotes name tables and columns and are not portable for text. A single quote inside a string is written twice: `'It''s ready'`.

## Comparison operators and AND/OR precedence
SQL compares with `=`, `<>` or `!=` for not equal, `<`, `>`, `<=`, `>=`. AND binds tighter than OR, so `a OR b AND c` means `a OR (b AND c)`; write parentheses to say what you mean. Comparing text with `=` can ignore case or spaces depending on the database collation.
