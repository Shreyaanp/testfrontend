import { useMemo } from 'react';
import { QRCodeSVG } from 'qrcode.react';
export default function QRCodeStage({ pairingToken, expiresIn }) {
    const displayToken = useMemo(() => pairingToken ?? 'waiting_token', [pairingToken]);
    return (<div className="overlay">
      <div className="overlay-card">
        <h1>Scan to pair</h1>
        <p>Use the mobile app to scan the QR code and continue.</p>
        <div className="qr-wrapper">
          <QRCodeSVG value={displayToken} size={240} includeMargin fgColor="#111" bgColor="#fff"/>
        </div>
        {typeof expiresIn === 'number' ? (<p className="expires">Expires in {Math.round(expiresIn)} seconds</p>) : null}
      </div>
    </div>);
}
