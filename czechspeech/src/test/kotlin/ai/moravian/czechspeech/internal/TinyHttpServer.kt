package ai.moravian.czechspeech.internal

import java.io.ByteArrayOutputStream
import java.io.Closeable
import java.io.IOException
import java.io.InputStream
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket

/**
 * A minimal HTTP/1.1 server on 127.0.0.1 for the download tests (the JDK's com.sun.net.httpserver is not on the
 * Android unit-test classpath). One request per connection.
 */
class TinyHttpServer(private val handler: (path: String, headers: Map<String, String>) -> Response) : Closeable {
    class Response(
        val code: Int,
        val headers: Map<String, String> = emptyMap(),
        val body: ByteArray = ByteArray(0),
        val offset: Int = 0,
    )

    private val socket = ServerSocket(0, 50, InetAddress.getByName("127.0.0.1"))
    val port: Int get() = socket.localPort

    init {
        Thread {
            while (!socket.isClosed) {
                val client = try {
                    socket.accept()
                } catch (e: IOException) {
                    break
                }
                Thread { serve(client) }.apply { isDaemon = true }.start()
            }
        }.apply { isDaemon = true }.start()
    }

    private fun serve(client: Socket) = client.use {
        val input = client.getInputStream()
        val requestLine = readLine(input) ?: return@use
        val headers = mutableMapOf<String, String>()
        while (true) {
            val line = readLine(input) ?: break
            if (line.isEmpty()) break
            headers[line.substringBefore(':').trim().lowercase()] = line.substringAfter(':').trim()
        }
        val path = requestLine.split(' ').getOrElse(1) { "/" }
        val response = handler(path, headers)
        val length = response.body.size - response.offset
        val head = buildString {
            append("HTTP/1.1 ${response.code} X\r\n")
            for ((k, v) in response.headers) append("$k: $v\r\n")
            append("Content-Length: $length\r\nConnection: close\r\n\r\n")
        }
        val out = client.getOutputStream()
        try {
            out.write(head.toByteArray(Charsets.US_ASCII))
            out.write(response.body, response.offset, length)
            out.flush()
        } catch (e: IOException) {
            // client went away
        }
    }

    private fun readLine(input: InputStream): String? {
        val buf = ByteArrayOutputStream()
        while (true) {
            val b = input.read()
            if (b < 0) return if (buf.size() == 0) null else buf.toString("US-ASCII")
            if (b == '\n'.code) return buf.toString("US-ASCII").trimEnd('\r')
            buf.write(b)
        }
    }

    override fun close() = socket.close()
}
