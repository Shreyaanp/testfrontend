interface InstructionStageProps {
  title: string
  subtitle?: string
}

export default function InstructionStage({ title, subtitle }: InstructionStageProps) {
  return (
    <div className="overlay">
      <div className="overlay-card">
        <h1>{title}</h1>
        {subtitle ? <p>{subtitle}</p> : null}
      </div>
    </div>
  )
}
