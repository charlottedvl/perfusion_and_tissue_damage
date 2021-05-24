from perfusion.API import API as Perfusion_API
from oxygen.API import API as Oxygen_API
import sys


from desist.eventhandler.api import API
from desist.eventhandler.eventhandler import event_handler


class Dispatch(API):
    """A wrapper API that dispatches to the right simulation.

    To choose the right part of the perfusion simulation, we dispatch on the
    type of the current event. The ``select_api`` routine returns the
    corresponding API implementation that can be evaluated.
    """
    def select_api(self):
        if self.current_model.get('type') == 'PERFUSION':
            api = event_handler(Perfusion_API)
            return api

        if self.current_model.get('type') == 'OXYGEN':
            api = event_handler(Oxygen_API)
            return api

        print(f"No API has been evaluated for model: `{self.current_model}`.")
        sys.exit(1)

    def event(self):
        api = self.select_api()
        return api()

    def example(self):
        api = self.select_api()
        return api()

    def test(self):
        api = self.select_api()
        return api()


if __name__ == "__main__":
    api = event_handler(Dispatch)
    api()
