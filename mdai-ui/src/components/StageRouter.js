import ErrorOverlay from './ErrorOverlay';
import IdleScreen from './IdleScreen';
import InstructionStage from './InstructionStage';
import QRCodeStage from './QRCodeStage';
export default function StageRouter({ state }) {
    if (state.matches('error')) {
        return <ErrorOverlay message={state.context.error ?? 'Unknown error'}/>;
    }
    if (state.matches('qr_display')) {
        return <QRCodeStage pairingToken={state.context.pairingToken} expiresIn={state.context.expiresIn}/>;
    }
    if (state.matches('pairing_request')) {
        return <InstructionStage title="Preparing session"/>;
    }
    if (state.matches('waiting_activation')) {
        return <InstructionStage title="Waiting for activation" subtitle="Complete the mobile step"/>;
    }
    if (state.matches('human_detect')) {
        return (<InstructionStage title="Center your face" subtitle="Move closer until your face fills the frame"/>);
    }
    if (state.matches('stabilizing')) {
        return <InstructionStage title="Hold steady" subtitle="Stay still for four seconds"/>;
    }
    if (state.matches('uploading')) {
        return <InstructionStage title="Uploading" subtitle="Please hold still"/>;
    }
    if (state.matches('waiting_ack')) {
        return <InstructionStage title="Processing" subtitle="This will take a moment"/>;
    }
    if (state.matches('complete')) {
        return <InstructionStage title="Completed" subtitle="You may step away"/>;
    }
    return <IdleScreen />;
}
