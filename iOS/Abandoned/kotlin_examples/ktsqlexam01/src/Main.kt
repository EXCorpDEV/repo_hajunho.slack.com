import java.sql.Connection
import java.sql.DriverManager

class App {

    companion object {
        @JvmStatic
        fun main(args: Array<String>) {
            Class.forName("org.sqlite.JDBC") // 꼭 필요함
            val conn: Connection = DriverManager.getConnection("jdbc:sqlite:sample.db")

            val stmt = conn.createStatement()
            stmt.executeUpdate("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)")
            stmt.executeUpdate("INSERT INTO users (name) VALUES ('홍길동')")
            val rs = stmt.executeQuery("SELECT * FROM users")

            while (rs.next()) {
                println("👤 id=${rs.getInt("id")}, name=${rs.getString("name")}")
            }

            conn.close()
        }
    }
}
