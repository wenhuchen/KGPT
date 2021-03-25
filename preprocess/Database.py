import sqlite3

class MyDatabase(object):

    def __init__(self, db_path, connect_each):
        self.db_path = db_path
        self.connect_each = connect_each
        if not connect_each:
            self.db = sqlite3.connect(db_path)
            self.cursor = self.db.cursor()
        self.tables = {}

    def create(self, create, table_name, keys):
        assert all([key[1] in ['INTEGER', 'TEXT'] for  key in keys])
        if create:
            assert table_name not in self.tables
            self.cursor.execute('''CREATE TABLE {}({})'''.format(table_name,
                        ", ".join(["{} {} {}".format(key[0], key[1], 'KEY' if i==0 else '')
                                for i, key in enumerate(keys)])))

            self.db.commit()
            query = "CREATE INDEX index_{} ON {}({})".format(keys[0][0], table_name, keys[0][0])
            self.cursor.execute(query)
            self.db.commit()
        self.tables[table_name] = [key[0] for key in keys]

    def insert(self, table_name, rows):
        assert table_name in self.tables
        assert all([len(self.tables[table_name])==len(row) for row in rows])
        query = '''INSERT INTO {}({}) VALUES({})'''.format(table_name,
                ", ".join(self.tables[table_name]),
                ",".join(["?" for _ in range(len(self.tables[table_name]))]))
        self.cursor.executemany(query, rows)

    def commit(self):
        self.db.commit()
        return self.rowcount_all()

    def rowcount_all(self):
        return ["{} {}".format(table_name, self.rowcount(table_name)) for table_name in  self.tables.keys()]

    def rowcount(self, table_name):
        self.cursor.execute("SELECT COUNT(*) FROM {}".format(table_name))
        return self.cursor.fetchone()[0]

    def fetch(self, table_name, key, value):
        #assert table_name in self.tables
        #assert key in self.tables[table_name]
        if self.connect_each:
            db = sqlite3.connect(self.db_path)
            cursor = db.cursor()
            cursor.execute('''SELECT * FROM {} where {}=?'''.format(table_name, key), (value,))
            rows = cursor.fetchall()
        else:
            self.cursor.execute('''SELECT * FROM {} where {}=?'''.format(table_name, key), (value,))
            rows = self.cursor.fetchall()
        return rows

    def close(self):
        self.db.close()
