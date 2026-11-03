import { Html, Head, Main, NextScript } from 'next/document'
import { GA_MEASUREMENT_ID } from '../lib/analytics'

// Google tag (gtag.js) — GA4 property shared with egagli.github.io.
// Inside egagli.github.io/maps/*.html the wrapper page already counted this
// visit, so skip the page view there and mark events as embedded.
// Placed in _document so it is written into the statically exported HTML.
const GA_INIT = `
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  window.gtag = gtag;
  gtag('js', new Date());
  var gaEmbedded = window.self !== window.top;
  gtag('set', { embedded: gaEmbedded });
  gtag('config', '${GA_MEASUREMENT_ID}', { send_page_view: !gaEmbedded });
`

export default function Document() {
  return (
    <Html lang="en">
      <Head>
        <script
          async
          src={`https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}`}
        />
        <script dangerouslySetInnerHTML={{ __html: GA_INIT }} />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  )
}
