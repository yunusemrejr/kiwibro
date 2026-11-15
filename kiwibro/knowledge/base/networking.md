# Networking and HTTP reference

## GET, read a resource
Asks the server for a resource, such as a web page or data. A GET request should not change anything on the server.
`curl https://example.com/api/items` sends a GET request.

## POST, create or submit
Sends data to the server, usually to create something or submit a form. Sending the same POST twice can create two copies.
`curl -X POST -d "name=test" https://example.com/form` sends a POST with form data.

## PUT, replace a resource
Sends data that replaces the whole resource at a URL. Sending the same PUT twice leaves the same final result.
`curl -X PUT -d "name=new" https://example.com/api/items/7` replaces item 7.

## PATCH, change part of a resource
Sends changes to part of a resource, such as one field, without sending the whole resource.
`curl -X PATCH -d "price=12" https://example.com/api/items/7` changes only the price of item 7.

## DELETE, remove a resource
Asks the server to delete the resource at a URL.
`curl -X DELETE https://example.com/api/items/7` asks the server to delete item 7.

## HEAD, headers only
Works like GET but the response has no body, only headers. It shows whether a resource exists and its size or type without downloading it.
`curl -I https://example.com` sends a HEAD request.

## OPTIONS, ask which methods are allowed
Asks the server which methods and options a URL supports. Browsers send it automatically before some requests that go to another site.
`curl -X OPTIONS -i https://example.com/api/items` prints the allowed methods in the response headers.

## 200 OK, request succeeded
The request worked and the response body holds the result.
A normal page load or API read returns 200.

## 201 Created, new resource made
The server created a new resource as a result of the request. The response often gives its address in a Location header.
A POST that adds a new item returns 201.

## 204 No Content, success with no body
The request worked and there is nothing to send back. Common after a delete or an update.
A DELETE that removes an item may return 204.

## 301 Moved Permanently, permanent redirect
The resource now lives at another address. Browsers and search engines switch to the new address.
An old URL that now points to a new one returns 301 with the new address in the Location header.

## 302 Found, temporary redirect
The resource is temporarily at another address. The original address should still be used later.
A login page that sends you to a sign-in screen can return 302.

## 304 Not Modified, cached copy is current
The browser asked for a resource it already has, and it has not changed. The browser uses its saved copy and no body is sent.
A conditional request with a matching ETag or Last-Modified date returns 304.

## 400 Bad Request
The server could not understand the request, usually because the data is malformed or a required part is missing.
A JSON body with a missing bracket or a wrong field name can return 400.

## 401 Unauthorized, login needed
The request has no valid login or credentials. Despite the name, it means not authenticated.
An API call without an access token, or with an expired one, returns 401.

## 403 Forbidden
The server understood the request and refuses it. Logging in does not help when the account has no permission for that resource.
A user who is logged in but may not view a private page gets 403.

## 404 Not Found
The server has no resource at that address.
A mistyped URL or a deleted page returns 404.

## 405 Method Not Allowed
The address exists but does not accept the HTTP method that was used. The response should list the allowed methods.
A POST sent to a URL that only accepts GET returns 405.

## 408 Request Timeout
The server waited too long for the complete request from the client and closed the connection.
A slow upload on a weak connection can end with 408.

## 409 Conflict
The request clashes with the current state of the resource, such as two edits to the same record at once.
Trying to create an account with a username that already exists can return 409.

## 410 Gone
The resource was removed on purpose and will not come back. Search engines drop it faster than for 404.
A page that was permanently taken down can return 410.

## 413 Payload Too Large, Content Too Large
The request body is bigger than the server accepts. Older documents call this Payload Too Large.
Uploading a file over the server's size limit returns 413.

## 415 Unsupported Media Type
The server does not accept the format of the data sent, as shown by the Content-Type header.
Sending XML to an endpoint that only accepts JSON returns 415.

## 418 I'm a teapot
A joke status code from an April 1998 proposal about coffee-making teapots. It is not used for real errors.
Some servers return it as a playful reply. Real clients should not depend on it.

## 422 Unprocessable Content
The request is well formed but its content is invalid, such as a date that does not exist. Older documents call it Unprocessable Entity.
A form with a valid format but an impossible value, such as a 31 February date, can return 422.

## 429 Too Many Requests, rate limit
The client sent too many requests in a short time. The response may include a Retry-After header that says when to try again.
A script that calls an API in a tight loop can be blocked with 429.

## 500 Internal Server Error
The server failed in a way it does not describe to the client. The fault is on the server side.
A bug in the server code that crashes on a request returns 500.

## 501 Not Implemented
The server does not support the method or feature needed to handle the request.
A server that does not understand a method it has never been built for returns 501.

## 502 Bad Gateway
A server acting as a gateway or proxy received an invalid reply from the server behind it.
A reverse proxy whose backend application has crashed can return 502.

## 503 Service Unavailable
The server cannot handle the request now, often because it is overloaded or down for maintenance. It may send a Retry-After header.
A site in maintenance mode returns 503.

## 504 Gateway Timeout
A gateway or proxy did not get a reply in time from the server behind it.
A proxy waiting on a slow database query can return 504.

## Ports 20 and 21, FTP file transfer
Port 21 carries FTP control commands, and port 20 carries the data in active mode. FTP sends passwords without encryption.
A plain FTP server listens on port 21. Use SFTP over port 22 or FTPS instead when security matters.

## Port 22, SSH
The default port for secure shell logins and secure file copies over SSH. It is encrypted.
`ssh alice@server.example.com` connects to port 22 unless another port is given.

## Port 23, Telnet
The default port for Telnet, an old remote login protocol that sends everything, including passwords, without encryption.
Avoid Telnet on any network you do not fully control. Use SSH on port 22 instead.

## Port 25, SMTP
The port that mail servers use to pass email to each other.
A mail server that receives mail for a domain listens on port 25.

## Port 53, DNS
The port for DNS queries. It usually uses UDP for short questions and TCP for large answers.
A DNS server answers queries on port 53.

## Ports 67 and 68, DHCP
Port 67 is used by the DHCP server and port 68 by the DHCP client. The client asks for an IP address and settings this way.
A router running a DHCP server listens on port 67, and your computer uses port 68 to receive its address.

## Port 80, HTTP
The default port for unencrypted web traffic. A plain http:// address uses it when no port is given.
`http://example.com` connects to port 80.

## Port 110, POP3
The port for POP3, a mail protocol that downloads mail to one device. Unencrypted on this port.
An email program that fetches mail with POP3 uses port 110, or port 995 for the encrypted version.

## Port 123, NTP
The port for NTP, the network time protocol that keeps computer clocks in step with time servers.
A computer syncing its clock with a time server uses UDP port 123.

## Port 143, IMAP
The port for IMAP, a mail protocol that keeps mail on the server and shows it on all devices.
An email program that reads mail with IMAP uses port 143, or port 993 for the encrypted version.

## Port 443, HTTPS
The default port for web traffic encrypted with TLS. An https:// address uses it when no port is given.
`https://example.com` connects to port 443.

## Port 465, SMTP over TLS
The port for sending mail with an encryption layer from the start of the connection.
A mail program set to SSL for outgoing mail often uses port 465.

## Port 587, mail submission
The port that mail programs use to send mail to their own mail server, usually with a login and STARTTLS encryption.
An outgoing mail server setting that asks for a login typically uses port 587.

## Port 993, IMAPS
IMAP over TLS, the encrypted form of IMAP. The connection is encrypted from the start.
A mail program set to SSL for incoming mail uses port 993.

## Port 995, POP3S
POP3 over TLS, the encrypted form of POP3.
A mail program set to SSL for POP3 mail uses port 995.

## Port 3306, MySQL and MariaDB
The default port for MySQL and MariaDB database servers.
An application that connects to a local MySQL server uses `127.0.0.1` on port 3306.

## Port 5432, PostgreSQL
The default port for PostgreSQL database servers.
`psql -h localhost -p 5432 -U alice mydb` connects to a local PostgreSQL server.

## Port 6379, Redis
The default port for Redis, an in-memory data store.
A Redis server started with its default settings listens on port 6379.

## Port 8080, alternative HTTP
A common port for web servers and development servers when port 80 is taken or needs no administrator rights. It is not the standard HTTP port.
`http://localhost:8080` opens a development server on this machine.

## Port 27017, MongoDB
The default port for MongoDB database servers.
A MongoDB server started with its default settings listens on port 27017.

## IP address
A number that identifies a device on a network so that traffic can reach it. Each device on the internet or a local network has one address for each network interface it uses.
`192.168.1.20` is an example of a local IPv4 address.

## IPv4 and IPv6
IPv4 addresses are 32-bit numbers written as four decimal groups, such as 192.0.2.10. IPv6 addresses are 128-bit numbers written as eight groups of hexadecimal digits, such as 2001:db8::10. IPv6 was made because IPv4 addresses ran out.
A device can have both kinds of address at the same time.

## Subnet mask
A pattern that splits an IP address into a network part and a host part. Devices with the same network part can reach each other directly.
255.255.255.0 means the first 24 bits are the network part, so 192.168.1.5 and 192.168.1.80 are on the same network.

## CIDR notation
A short way to write an address and its subnet size, using a slash and the number of network bits. The bigger the number, the smaller the network.
`192.168.1.0/24` means the network 192.168.1.0 with 256 addresses. `10.0.0.0/8` is a much larger network.

## Private address ranges
Blocks of IP addresses reserved for use inside local networks. Packets to them are not routed on the public internet.
The ranges are 10.0.0.0/8, 172.16.0.0/12 and 192.168.0.0/16. Home routers usually hand out addresses from 192.168.x.x.

## DNS, domain name system
The system that turns names such as example.com into IP addresses. Your computer asks a DNS server, which answers from its records or asks other servers.
`dig example.com A +short` asks DNS for the IPv4 address of a name.

## A record, IPv4 address for a name
A DNS record that maps a name to an IPv4 address.
A record for `example.com` with the value 192.0.2.10 sends visitors to that address.

## AAAA record, IPv6 address for a name
A DNS record that maps a name to an IPv6 address.
A record for `example.com` with an IPv6 value lets IPv6 clients reach the site.

## CNAME record, alias for another name
A DNS record that says one name is an alias for another name. The DNS server then looks up the target name.
`www.example.com` as a CNAME for `example.com` makes both names lead to the same address.

## MX record, mail server for a domain
A DNS record that names the mail servers that accept email for a domain. Each record has a priority number, and lower numbers are tried first.
An MX record for `example.com` pointing at `mail.example.com` sends its email there.

## TXT record, text data for a name
A DNS record that holds free text. Many services use it to prove domain ownership and to publish email policies such as SPF.
A TXT record can hold a verification code that a service asks you to add to your domain.

## NS record, name servers for a domain
A DNS record that lists the servers which are authoritative for a domain. They answer for all names in it.
An NS record for `example.com` names the servers that hold its records.

## DHCP, automatic network settings
A protocol that gives a device its IP address, subnet mask, default gateway and DNS servers when it joins a network, so nobody has to type them in.
Your home router runs DHCP, and each device receives an address from its pool when it connects.

## NAT, network address translation
A router rewrites addresses so many devices on a private network can share one public IP address. It keeps track of which connection belongs to which device.
When your laptop loads a web page, the router sends it out with its public address and passes the reply back to the laptop.

## TCP and UDP
TCP is a connection-based protocol that delivers data in order and retries lost parts. UDP sends independent packets with no delivery guarantee, which makes it faster and lighter.
Web pages, email and SSH use TCP. Live video calls and DNS queries often use UDP.

## TLS
The encryption layer that protects a network connection, so that others cannot read or change the data in transit. It also proves the server's identity with a certificate. It is the successor of SSL.
HTTPS, SMTPS and IMAPS are TCP connections protected with TLS.

## HTTPS
HTTP sent over a TLS connection. The address starts with https:// and the browser shows a lock.
Logins and payments on websites should always use HTTPS.

## localhost
The name your own computer uses for itself. It always points to the loopback address.
`http://localhost:8080` opens a server that runs on your own machine.

## Loopback address 127.0.0.1
The IPv4 address that means this computer. Traffic to it never leaves the machine. The whole 127.0.0.0/8 range is loopback, and IPv6 uses ::1.
A server that listens on 127.0.0.1 can only be reached from the same computer.

## Firewall
A set of rules that allows or blocks network traffic based on address, port and direction. It protects a machine or a network from unwanted connections.
A firewall that allows only port 22 and port 443 lets SSH and HTTPS in and blocks everything else.

## VPN, virtual private network
An encrypted tunnel between your device and another network. Traffic inside the tunnel appears to come from the other network.
A company VPN lets you reach its internal servers from home, as if you were in the office.

## Proxy
A server that sends requests on behalf of clients. Clients connect to the proxy, and the proxy connects to the destination.
A company proxy can record and filter web traffic, and a reverse proxy in front of a web app forwards visitors to the app.

## MAC address
A hardware identifier of a network interface, written as six pairs of hexadecimal digits such as 00:1A:2B:3C:4D:5E. It is used on the local network segment, not across the internet.
`ip link` shows the MAC address of each interface on Linux.

## Latency
The time a message takes to travel from one point to another and back, usually measured in milliseconds. Low latency means quick responses.
`ping example.com` shows the round-trip time in milliseconds.

## Bandwidth
The maximum amount of data a connection can carry per second, measured in bits per second. It is the size of the pipe, not how fast the first bit arrives.
A 100 Mbit/s connection carries at most about 100 million bits each second, however low the latency.

## URL parts
A URL is made of a scheme, a host, an optional port, a path, an optional query and an optional fragment.
In `https://example.com:443/docs/page?lang=en#top`, https is the scheme, example.com the host, 443 the port, /docs/page the path, lang=en the query, and top the fragment.

## Cookie
A small piece of text that a server asks the browser to store and send back with later requests to the same site. Cookies carry sessions and preferences.
A login session cookie lets a site recognise you on the next page without asking for your password again.

## CORS, cross-origin resource sharing
A browser rule that blocks a web page from reading responses from another site, unless that site allows it with CORS headers. The server decides which origins may read its responses.
A page on one site that calls an API on another site needs the API to send an Access-Control-Allow-Origin header.

## REST
A style of web API in which each resource has its own URL, and HTTP methods such as GET, POST, PUT and DELETE act on it. Responses usually carry JSON.
`GET /api/items/7` reads item 7, and `DELETE /api/items/7` removes it.

## WebSocket
A connection that starts as HTTP and then stays open, so the server and the browser can send messages to each other at any time. It suits chat and live updates.
A chat application opens a WebSocket so that new messages arrive without the page asking again.
