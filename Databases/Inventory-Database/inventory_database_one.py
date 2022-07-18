import psycopg2


def table():
    con=psycopg2.connect("dbname= 'database1' user = 'postgres' password= '' host = 'localhost' port = '1234' ")
    cur=con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS inventory (item TEXT, quantity INTEGER, price REAL)")
    con.commit()
    con.close()

def insert(item, quantity, price):
    con = psycopg2.connect("dbname= 'database1' user = 'postgres' password= '' host = 'localhost' port = '1234' ")
    cur = con.cursor()
#    cur.execute("INSERT INTO inventory VALUES('%s','%s','%s')" % (item,quantity,price))
    cur.execute("INSERT INTO inventory VALUES(%s,%s,%s)", (item,quantity,price))
    con.commit()
    con.close()

def view():
    con = psycopg2.connect("dbname= 'database1' user = 'postgres' password= '' host = 'localhost' port = '1234' ")
    cur = con.cursor()
    cur.execute("SELECT * FROM inventory")
    rows = cur.fetchall()
    con.close()
    return rows

def delete(item):
    con = psycopg2.connect("dbname= 'database1' user = 'postgres' password= '' host = 'localhost' port = '1234' ")
    cur = con.cursor()
    cur.execute("DELETE FROM inventory WHERE item = %s", (item,))
    con.commit()
    con.close()

def update(quantity,price,item):
    con = psycopg2.connect("dbname= 'database1' user = 'postgres' password= '' host = 'localhost' port = '1234' ")
    cur = con.cursor()
    cur.execute("UPDATE inventory SET quantity=%s, price=%s WHERE item =%s",(quantity,price,item))
    con.commit()
    con.close()

#table()
insert("coffee cup",2,5)
print(view())
