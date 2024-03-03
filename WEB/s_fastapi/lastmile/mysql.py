import pymysql.cursors

# Connect to the database
connection = pymysql.connect(host='db-a9lnl-kr.vpc-pub-cdb.ntruss.com',
                             user='sodoc_access',
                             password='t?s7=tKvaSC8vW?F{kMu',
                             database='sodoc',
                             cursorclass=pymysql.cursors.DictCursor)

with connection:
    with connection.cursor() as cursor:
        # Create a new record
        sql = "SELECT DISTINCT A.BADDR FROM t_b2b A LEFT JOIN t_delivery B ON A.B2BID=B.BNAME where A.B2BID=B.BNAME limit 1"
        cursor.execute(sql)
        ret = cursor.fetchone()
        print(ret)

    # connection is not autocommit by default. So you must commit to save
    # your changes.
    connection.commit()

    # with connection.cursor() as cursor:
    #     # Read a single record
    #     sql = "SELECT `id`, `password` FROM `users` WHERE `email`=%s"
    #     cursor.execute(sql, ('webmaster@python.org',))
    #     result = cursor.fetchone()
    #     print(result)