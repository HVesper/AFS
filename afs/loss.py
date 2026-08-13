import torch


class AFSLoss:
    """Dense stop-gradient velocity matching over Student-visited states."""

    def __call__(self, student_velocities, teacher_velocities, weights=None):
        if len(student_velocities) != len(teacher_velocities) or not student_velocities:
            raise ValueError("Student and Teacher velocity lists must be non-empty and aligned")
        terms = []
        for index, (student, teacher) in enumerate(zip(student_velocities, teacher_velocities)):
            term = (student - teacher.detach()).square().mean()
            if weights is not None:
                term = term * weights[index]
            terms.append(term)
        return torch.stack(terms).mean()
