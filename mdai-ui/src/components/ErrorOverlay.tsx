interface ErrorOverlayProps {
  message: string
}

export default function ErrorOverlay({ message }: ErrorOverlayProps) {
  return (
    <div className="overlay error">
      <div className="overlay-card">
        <h1>Something went wrong</h1>
        <p>{message}</p>
      </div>
    </div>
  )
}
