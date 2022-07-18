import sqlite3


def table():
    con=sqlite3.connect("data.db")
    cur=con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS inventory (item TEXT, quantity INTEGER, price REAL)")
    con.commit()
    con.close()

def insert(item, quantity, price):
    con = sqlite3.connect("data.db")
    cur = con.cursor()
    cur.execute("INSERT INTO inventory VALUES(?,?,?)",(item,quantity,price))
    con.commit()
    con.close()

def view():
    con = sqlite3.connect("data.db")
    cur = con.cursor()
    cur.execute("SELECT * FROM inventory")
    rows = cur.fetchall()
    con.close()
    return rows

def delete(item):
    con = sqlite3.connect("data.db")
    cur = con.cursor()
    cur.execute("DELETE FROM inventory WHERE item =?", (item,))
    con.commit()
    con.close()

def update(quantity,price,item):
    con = sqlite3.connect("data.db")
    cur = con.cursor()
    cur.execute("UPDATE inventory SET quantity=?, price=? WHERE item =?",(quantity,price,item))
    con.commit()
    con.close()

update(2,5,"coffee cup")
print(view())
