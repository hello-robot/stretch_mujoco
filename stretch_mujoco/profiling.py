yappi_profiling = {}
def yappi_profile(tag:str, is_wall_clock:bool = True):
    """
    Call once with a tag to start profiling. Call again to stop profiling.

    Wall vs CPU clock types: https://github.com/sumerc/yappi/blob/master/doc/clock_types.md 
    """
    global yappi_profiling

    import yappi
    import sys

    file_handler = sys.stdout
    
    if not tag in yappi_profiling:
        yappi.set_clock_type("wall" if is_wall_clock else "cpu")
        return yappi.start()
        
    yappi.stop()


    print("---------------get_thread_stats------------------\n\n")
    # retrieve thread stats by their thread id (given by yappi)
    threads = yappi.get_thread_stats()
    for thread in threads:
        print(
            f"\nFunction stats for {thread.name} ({thread.id})\n"
        )  # it is the Thread.__class__.__name__
        stats = yappi.get_func_stats(ctx_id=thread.id)
        stats.print_all(file_handler)


    print("---------------get_func_stats------------------\n\n")
    # Stats sorted by total time
    stats = yappi.get_func_stats().sort(
            sort_type='totaltime', sort_order='desc') 
    # returns all stats with sorting applied 
    stats.print_all(file_handler)
