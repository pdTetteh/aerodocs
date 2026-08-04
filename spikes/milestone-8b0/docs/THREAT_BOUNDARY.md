# Threat boundary

```text
Untrusted browser/API client
           |
           | public HTTPS only
           v
 stock Caddy 2.11.4
   | exact idp host --------> Keycloak 26.7.0
   |                           loopback-only HTTP listener
   |
   | /oauth2/* -------------> OAuth2 Proxy 7.15.3
   |                           static success upstream only
   |
   | bodyless /oauth2/auth --> OAuth2 Proxy
   |                           returns auth envelope only
   |
   +------------------------> mock AeroDocs
                               application request/response data plane
```

Only Caddy is externally reachable. The IdP and AeroDocs hostnames have
separate Caddy site blocks and share no authentication routing rules. Keycloak,
OAuth2 Proxy, and the mock bind to loopback. OAuth2 Proxy cannot connect to the
mock upstream and receives no AeroDocs body or response.

Caddy removes every incoming internal identity header before routing. Direct
bearer requests retain their raw Authorization representation, lose cookies,
skip OAuth2 Proxy, and receive no gateway token. Browser-session requests lose
Authorization and all client-supplied identity headers; after a successful
bodyless check, Caddy may construct only
`X-AeroDocs-Gateway-Access-Token` from
`X-Auth-Request-Access-Token`.

The spike treats all credentials as toxic data. Probes may record only counts,
byte lengths, booleans, status codes, and bounded synthetic identifiers.
Cryptographic hashes are retained only for public binaries, immutable images,
stock-module inventories, and non-secret configuration—not credentials.
