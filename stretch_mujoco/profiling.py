yappi_profiling = {}
def yappi_profile(tag:str, is_wall_clock:bool = True):
    """
    Call once with a tag to start profiling. Call again to stop profiling.

    Wall vs CPU clock types: https://github.com/sumerc/yappi/blob/master/doc/clock_types.md 
    """
    global yappi_profiling

    import yappi
    
    if not tag in yappi_profiling:
        yappi.set_clock_type("wall" if is_wall_clock else "cpu")
        return yappi.start()
        
    yappi.stop()

    # retrieve thread stats by their thread id (given by yappi)
    threads = yappi.get_thread_stats()
    for thread in threads:
        print(
            "Function stats for (%s) (%d)" % (thread.name, thread.id)
        )  # it is the Thread.__class__.__name__
        yappi.get_func_stats(ctx_id=thread.id).print_all()


    print("---------------------------------\n\n")
    import sys
    from yappi import get_func_stats, COLUMNS_FUNCSTATS
    # Stats sorted by total time
    stats = get_func_stats().sort(
            sort_type='totaltime', sort_order='desc') 
    # returns all stats with sorting applied 
    stats.print_all(sys.stdout)
